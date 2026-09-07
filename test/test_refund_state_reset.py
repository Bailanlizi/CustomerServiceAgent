# test/test_refund_state_reset.py
"""
P1 修复回归测试：退款 FSM 终态短路与空回答兜底。

覆盖架构回归:
  1. 上一笔退款留下 refund_submitted_id 终态时，新一轮"我要退款"
     必须清空终态字段，FSM 才能重新走询问/收集流程。
  2. 纯"查进度"轮次（无退款意图关键词）不能误清终态，避免破坏后续提交。
  3. LLM 返回 AIMessage 但 content 为空时，三个追问节点的兜底必须返回非空。
  4. 子图入口直接路由到 DONE/REJECTED 时，必须写入兜底 answer。
  5. chat.py 占位回答按 workflow_stage 派生合理文本。
"""
from types import SimpleNamespace

import pytest

from app.api.v1.chat import _refund_placeholder_answer
from app.conversation.state_manager import (
    ConversationStateManager,
    SlotExtraction,
    default_working_memory,
)
from app.graph.workflows.refund import RefundStage


class StubStructuredLLM:
    """最小可用的 structured LLM 桩,避免网络调用。"""

    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    async def ainvoke(self, _messages):
        if self.error:
            raise self.error
        return self.value


def session(memory=None, summary=None):
    return SimpleNamespace(
        working_memory_json=memory or default_working_memory(),
        conversation_summary=summary,
    )


# ============================================================
# A. 终态短路修复: 新一轮退款必须清掉 refund_submitted_id
# ============================================================
@pytest.mark.asyncio
async def test_prepare_turn_clears_terminal_state_on_new_refund_intent():
    """用户说"我要退款"且工作记忆残留上一笔 submitted_id,prepare_turn
    必须清空终态字段,FSM 才能走询问/收集流程,而不是直接 DONE 空回答。"""
    memory = default_working_memory()
    memory.update({
        "active_domain": "REFUND",
        "active_order_id": 3,
        "active_order_sn": "SN20240003",
        "workflow_stage": RefundStage.DONE.value,
        "user_confirmed": True,
        "last_tool_result": {"eligibility_checked": True, "eligibility_passed": True},
        "collected_slots": {
            "refund_submitted_id": 42,
            "user_confirmed": True,
            "refund_reason": "尺码不合适",
        },
    })
    extractor = StubStructuredLLM(
        SlotExtraction(domain="REFUND", explicit_new_goal=True, conversation_goal="申请退款")
    )
    manager = ConversationStateManager(extractor=extractor, summarizer=StubStructuredLLM())

    result = await manager.prepare_turn(session(memory), "我要退款")

    slots = result["collected_slots"]
    assert "refund_submitted_id" not in slots, "submitted_id 必须清空"
    assert "user_confirmed" not in slots, "user_confirmed 必须清空,强制重新确认"
    assert result["user_confirmed"] is None
    assert result["workflow_stage"] is None, "workflow_stage 必须清空,让路由重新计算"
    assert result["last_tool_result"] is None
    # 用户的退款原因 / 活跃订单应当保留,避免重复追问
    assert result["active_order_sn"] == "SN20240003"
    assert slots.get("refund_reason") == "尺码不合适"
    # 重新进入 REFUND 域后,FSM 应该要求确认(因为只剩退款原因)
    assert result["active_domain"] == "REFUND"


@pytest.mark.asyncio
async def test_prepare_turn_keeps_terminal_state_on_status_query():
    """用户纯问退款进度时,不能误清 submitted_id;终态标记必须保留,
    这样状态查询轮次不会被误判为新一轮退款。"""
    memory = default_working_memory()
    memory.update({
        "active_domain": "REFUND",
        "active_order_id": 3,
        "active_order_sn": "SN20240003",
        "workflow_stage": RefundStage.DONE.value,
        "collected_slots": {"refund_submitted_id": 42, "refund_reason": "尺码不合适"},
    })
    # 显式新目标=False,且消息不含退款意图关键词
    extractor = StubStructuredLLM(SlotExtraction(domain="REFUND", explicit_new_goal=False))
    manager = ConversationStateManager(extractor=extractor, summarizer=StubStructuredLLM())

    result = await manager.prepare_turn(session(memory), "我的退款申请进度怎么样了")

    slots = result["collected_slots"]
    assert slots.get("refund_submitted_id") == 42, "查进度轮次不能清掉 submitted_id"
    assert result["workflow_stage"] == RefundStage.DONE.value


@pytest.mark.asyncio
async def test_prepare_turn_clears_when_keyword_present_without_explicit_flag():
    """当 LLM 提取器没设 explicit_new_goal 但消息含退款关键词时,
    兜底路径仍应清空终态,避免 LLM 抽风导致 FSM 持续短路。"""
    memory = default_working_memory()
    memory.update({
        "active_domain": "REFUND",
        "active_order_id": 3,
        "collected_slots": {"refund_submitted_id": 99},
    })
    extractor = StubStructuredLLM(SlotExtraction(domain="REFUND", explicit_new_goal=False))
    manager = ConversationStateManager(extractor=extractor, summarizer=StubStructuredLLM())

    result = await manager.prepare_turn(session(memory), "我要退货")

    assert "refund_submitted_id" not in result["collected_slots"]


@pytest.mark.asyncio
async def test_prepare_turn_keeps_state_when_no_terminal_marker():
    """没有 submitted_id 时,不应该做任何清空(无副作用)。"""
    memory = default_working_memory()
    memory.update({"active_domain": "REFUND", "active_order_sn": "SN20240003"})
    extractor = StubStructuredLLM(
        SlotExtraction(domain="REFUND", explicit_new_goal=True, conversation_goal="申请退款")
    )
    manager = ConversationStateManager(extractor=extractor, summarizer=StubStructuredLLM())

    result = await manager.prepare_turn(session(memory), "我要退款")

    # 没有 submitted_id,清理逻辑不触发,collected_slots 保持原样(空或仅含既有键)
    assert "refund_submitted_id" not in result["collected_slots"]
    assert result["active_order_sn"] == "SN20240003"


# ============================================================
# B. 子图入口兜底答案: 终态路由必须有非空 answer
# ============================================================
@pytest.mark.asyncio
async def test_refund_subgraph_entry_emits_placeholder_for_done():
    """子图入口路由到 DONE 时,必须写入 answer,避免空回答被前端只渲染成
    PROCESSING 状态文本。"""
    from app.graph.workflows.refund import refund_subgraph_entry

    state = {
        **default_working_memory(),
        "active_order_id": 3,
        "active_order_sn": "SN20240003",
        "collected_slots": {"refund_submitted_id": 7, "refund_reason": "尺码不合适"},
    }
    payload = await refund_subgraph_entry(state)
    assert payload["workflow_stage"] == RefundStage.DONE.value
    assert payload["answer"], "DONE 终态必须有兜底 answer"
    assert "#7" in payload["answer"]


@pytest.mark.asyncio
async def test_refund_subgraph_entry_emits_placeholder_for_rejected():
    """REJECTED 终态也必须有兜底 answer。"""
    from app.graph.workflows.refund import refund_subgraph_entry

    state = {
        **default_working_memory(),
        "active_order_id": 3,
        "active_order_sn": "SN20240003",
        "collected_slots": {"refund_reason": "已发货"},
        "last_tool_result": {
            "eligibility_checked": True,
            "eligibility_passed": False,
            "eligibility_message": "已发货超过7天",
        },
    }
    payload = await refund_subgraph_entry(state)
    assert payload["workflow_stage"] == RefundStage.REJECTED.value
    assert payload["answer"]


@pytest.mark.asyncio
async def test_refund_subgraph_entry_does_not_emit_answer_for_non_terminal():
    """非终态路由(IDENTIFY_ORDER 等)不主动写 answer,由对应节点写。"""
    from app.graph.workflows.refund import refund_subgraph_entry

    state = {**default_working_memory()}
    payload = await refund_subgraph_entry(state)
    assert payload["workflow_stage"] == RefundStage.IDENTIFY_ORDER.value
    assert "answer" not in payload


# ============================================================
# C. 话术节点确定性模板（P0：原 LLM 流式追问已改为固定模板，不再调模型）
# ============================================================
@pytest.mark.asyncio
async def test_identify_order_uses_template_when_no_order_sn():
    """缺订单号时返回固定追问话术，且 messages 与 answer 一致。"""
    from app.graph.workflows import refund as refund_mod

    state = {"question": "我要退款", "messages": []}
    result = await refund_mod.node_identify_order(state)
    assert result["answer"], "必须返回追问话术"
    assert "订单号" in result["answer"]
    assert result["messages"][0].content == result["answer"]


@pytest.mark.asyncio
async def test_collect_reason_uses_template_when_reason_missing():
    """订单已确认但缺原因时返回固定追问话术。"""
    from app.graph.workflows import refund as refund_mod

    state = {
        "question": "尺码不合适",
        "active_order_sn": "SN20240003",
        "collected_slots": {},
        "messages": [],
    }
    result = await refund_mod.node_collect_reason(state)
    assert result["answer"], "必须返回追问话术"
    assert "退货原因" in result["answer"] or "原因" in result["answer"]
    assert result["messages"][0].content == result["answer"]


@pytest.mark.asyncio
async def test_await_confirmation_uses_template_with_slots():
    """确认话术由订单号/原因/资格结果插值生成，结尾提示「确认提交」。"""
    from app.graph.workflows import refund as refund_mod

    state = {
        "question": "确认提交",
        "active_order_sn": "SN20240003",
        "collected_slots": {"refund_reason": "尺码不合适"},
        "last_tool_result": {"eligibility_message": "✅ 符合退货条件"},
        "messages": [],
    }
    result = await refund_mod.node_await_confirmation(state)
    assert result["answer"], "必须返回确认话术"
    assert "确认提交" in result["answer"]
    assert "SN20240003" in result["answer"]
    assert "尺码不合适" in result["answer"]
    assert result["messages"][0].content == result["answer"]


# ============================================================
# D. 占位回答派生
# ============================================================
def test_refund_placeholder_answer_done():
    answer = _refund_placeholder_answer({"workflow_stage": "DONE"})
    assert "退款" in answer


def test_refund_placeholder_answer_waiting_confirmation():
    answer = _refund_placeholder_answer({"workflow_stage": "WAITING_CONFIRMATION"})
    assert "确认提交" in answer


def test_refund_placeholder_answer_unknown_stage():
    answer = _refund_placeholder_answer({"workflow_stage": "FUTURE_STAGE"})
    assert answer  # 任何非 None 都通过