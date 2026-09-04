from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.conversation.state_manager import (
    SUMMARY_PINNED_SN_PREFIX,
    SUMMARY_RETAIN_TURNS,
    SUMMARY_TRIGGER_TURNS,
    ConversationStateManager,
    ConversationSummary,
    SlotExtraction,
    default_working_memory,
)


class StubStructuredLLM:
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


@pytest.mark.asyncio
async def test_refund_followups_merge_confirmed_slots_without_reasking():
    extractor = StubStructuredLLM(SlotExtraction(domain="REFUND", conversation_goal="申请退款"))
    manager = ConversationStateManager(extractor=extractor, summarizer=StubStructuredLLM())
    first = await manager.prepare_turn(session(), "我要退款")
    assert first["active_domain"] == "REFUND"
    assert first["pending_slots"] == ["order_sn", "refund_reason"]

    extractor.value = SlotExtraction(domain="OTHER")
    second = await manager.prepare_turn(session(first), "SN20240001")
    assert second["active_domain"] == "REFUND"
    assert second["active_order_sn"] == "SN20240001"
    assert second["pending_slots"] == ["refund_reason"]

    extractor.value = SlotExtraction(domain="REFUND", refund_reason="尺码不合适")
    third = await manager.prepare_turn(session(second), "尺码不合适")
    assert third["collected_slots"]["refund_reason"] == "尺码不合适"
    assert third["pending_slots"] == []


@pytest.mark.asyncio
async def test_new_explicit_order_clears_verified_order_id():
    memory = default_working_memory()
    memory.update({"active_domain": "REFUND", "active_order_id": 7, "active_order_sn": "SN1"})
    manager = ConversationStateManager(
        extractor=StubStructuredLLM(SlotExtraction(domain="OTHER")),
        summarizer=StubStructuredLLM(),
    )
    result = await manager.prepare_turn(session(memory), "换成 SN20240002")
    assert result["active_order_id"] is None
    assert result["active_order_sn"] == "SN20240002"


@pytest.mark.asyncio
async def test_summary_pins_active_order_sn_to_first_line():
    """即使 LLM 没在已确认事实里写出 SN，_summarize 也要主动把 SN 置顶到第一行。

    背景：P1 修复把历史裁切从 12 触发 / 3 保留收紧到 6 / 5，旧 SN 可能不在保留窗口
    里；唯一稳定的"当前 case 订单号"来源就是 working memory。摘要必须把这条事实
    强制置顶，让退款 Agent 在下一轮仍能读到。
    """
    manager = ConversationStateManager(
        extractor=StubStructuredLLM(),
        # LLM 故意不写 SN，模拟长对话压缩后模型已遗忘的场景
        summarizer=StubStructuredLLM(
            ConversationSummary(confirmed_facts=["用户咨询了运费政策"])
        ),
    )
    messages = [HumanMessage(content=str(i)) for i in range(SUMMARY_TRIGGER_TURNS)]
    summary = await manager._summarize(
        messages,
        {"active_order_id": 9, "active_order_sn": "SN20240001"},
        None,
    )
    assert summary is not None
    # 置顶前缀必须在第一行
    first_line = summary.split("\n", 1)[0]
    assert first_line == f"{SUMMARY_PINNED_SN_PREFIX}SN20240001"
    # SN 必须出现在摘要里（required 校验依然生效）
    assert "SN20240001" in summary


@pytest.mark.asyncio
async def test_summary_without_active_order_sn_skips_pinning():
    """没有 active_order_sn 时不强制置顶，避免对闲聊会话添加无意义前缀。"""
    manager = ConversationStateManager(
        extractor=StubStructuredLLM(),
        summarizer=StubStructuredLLM(
            ConversationSummary(confirmed_facts=["用户咨询了运费政策"])
        ),
    )
    messages = [HumanMessage(content=str(i)) for i in range(SUMMARY_TRIGGER_TURNS)]
    summary = await manager._summarize(messages, {}, None)
    assert summary is not None
    assert SUMMARY_PINNED_SN_PREFIX not in summary


@pytest.mark.asyncio
async def test_summary_does_not_duplicate_pin_when_llm_already_writes_sn():
    """LLM 自己写出 SN 时不应重复拼接置顶行。"""
    manager = ConversationStateManager(
        extractor=StubStructuredLLM(),
        summarizer=StubStructuredLLM(
            ConversationSummary(confirmed_facts=["订单 SN20240001 已发货"])
        ),
    )
    messages = [HumanMessage(content=str(i)) for i in range(SUMMARY_TRIGGER_TURNS)]
    summary = await manager._summarize(
        messages,
        {"active_order_id": 9, "active_order_sn": "SN20240001"},
        None,
    )
    assert summary is not None
    # 仍要置顶，但只出现一次
    assert summary.count(f"{SUMMARY_PINNED_SN_PREFIX}SN20240001") == 1


def test_compaction_retains_last_five_complete_turns():
    """P1 修复: 摘要保留窗口由 3 轮放宽到 5 轮，避免裁掉跨意图中间事实。
    messages 数等于 SUMMARY_RETAIN_TURNS 时不裁切；超过才从倒数第 N 条 HumanMessage
    开始保留。
    """
    messages = []
    for i in range(8):
        messages.extend([HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")])
    retained = ConversationStateManager.retained_messages(messages)
    assert [item.content for item in retained] == [
        "q3", "a3", "q4", "a4", "q5", "a5", "q6", "a6", "q7", "a7",
    ]


# ============================================================
# P1 修复回归测试（修复 5/6）
# - 修复 5：AgentState.history 字段已删除，避免与 messages 双重历史路径
# - 修复 6：摘要阈值由 12/3 收紧到 6/5，并强制置顶 active_order_sn
# ============================================================


def test_agent_state_no_longer_declares_history_field():
    """AgentState 必须彻底删除 history 字段，避免与 messages 形成双重历史路径。

    旧实现里 chat.py 传 `"history": []` 硬编码空列表、nodes.py 从不读取 state["history"]，
    其他开发者会以为它有用。本次回归测试用 Annotated[...] 探测：若有人把字段加回来，
    这里立即失败。
    """
    from app.graph.state import AgentState

    assert "history" not in AgentState.__annotations__, (
        "AgentState.history 已被删除，禁止通过 'history' 字段重新引入；"
        "对话历史请使用 AgentState.messages（由 LangGraph checkpoint 维护）。"
    )


def test_summary_thresholds_match_average_support_window():
    """摘要阈值必须在 6/5 上锁住，避免有人无意中改回 12/3。

    客服平均对话 4-8 轮，12 触发偏晚；3 保留容易裁掉跨意图中间事实。
    """
    assert SUMMARY_TRIGGER_TURNS == 6
    assert SUMMARY_RETAIN_TURNS == 5


@pytest.mark.asyncio
async def test_explicit_new_goal_with_concrete_domain_overrides_previous_domain():
    """explicit_new_goal=True 时，明确领域分类必须无脑切换，覆盖兜底逻辑。"""
    memory = default_working_memory()
    memory.update({"active_domain": "POLICY"})
    extractor = StubStructuredLLM(
        SlotExtraction(
            domain="REFUND",
            explicit_new_goal=True,
            refund_reason="质量问题",
            refund_reason_category="QUALITY_ISSUE",
        )
    )
    manager = ConversationStateManager(extractor=extractor, summarizer=StubStructuredLLM())
    result = await manager.prepare_turn(session(memory), "我要退款")
    assert result["active_domain"] == "REFUND", (
        "显式新目标应覆盖 previous_domain，不能被截胡到 POLICY"
    )


@pytest.mark.asyncio
async def test_explicit_new_goal_with_other_falls_back_to_previous_domain():
    """explicit_new_goal=True 但 LLM 兜底为 OTHER 时，保留 previous_domain 避免误切。"""
    memory = default_working_memory()
    memory.update({"active_domain": "ORDER"})
    extractor = StubStructuredLLM(SlotExtraction(domain="OTHER", explicit_new_goal=True))
    manager = ConversationStateManager(extractor=extractor, summarizer=StubStructuredLLM())
    result = await manager.prepare_turn(session(memory), "嗯")
    assert result["active_domain"] == "ORDER", (
        "OTHER 兜底不应强行切换领域"
    )


@pytest.mark.asyncio
async def test_refund_reason_category_persists_into_working_memory():
    """SlotExtraction.refund_reason_category 必须写入 working memory，供提交工具通过
    InjectedState 直接消费，避免 LLM 在工具调用时再次做 free-text → enum 映射。"""
    extractor = StubStructuredLLM(
        SlotExtraction(
            domain="REFUND",
            explicit_new_goal=True,
            refund_reason="尺码不合适",
            refund_reason_category="SIZE_NOT_FIT",
        )
    )
    manager = ConversationStateManager(extractor=extractor, summarizer=StubStructuredLLM())
    result = await manager.prepare_turn(session(), "尺码不合适")
    assert result["refund_reason_category"] == "SIZE_NOT_FIT"
    assert result["collected_slots"]["refund_reason"] == "尺码不合适"


def test_default_working_memory_includes_reason_category_key():
    """refund_reason_category 必须出现在 default_working_memory 和 MEMORY_KEYS 中，
    否则 persist_turn 不会写入数据库。"""
    from app.conversation.state_manager import MEMORY_KEYS

    assert "refund_reason_category" in default_working_memory()
    assert "refund_reason_category" in MEMORY_KEYS


@pytest.mark.asyncio
async def test_invalid_reason_category_value_is_rejected_by_schema():
    """SlotExtraction.refund_reason_category 必须严格枚举，非法值不能落入工作记忆
    （避免 LLM 输出 '质量问题' 这种 free-text 时污染下游 enum）。"""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SlotExtraction(domain="REFUND", refund_reason_category="质量问题")
