# test/test_refund_workflow_fsm.py
"""P1 验收回归测试：RefundWorkflow FSM 6 阶段显式状态机。

覆盖 architecture-update.md 第 222–224 行的三项验收：
  1. 同一退款流程中每个槽位最多主动询问一次；
  2. 未完成确认不能提交；
  3. 已提交申请不会再次创建。

外加：route_refund_stage 阶段路由 6 阶段覆盖 + prepare_turn user_confirmed 双向同步。
"""
from typing import Any

import pytest

from app.conversation.state_manager import (
    ConversationStateManager,
    default_working_memory,
)
from app.graph.tool_registry import tool_registry
from app.graph.workflows.refund import (
    RefundStage,
    route_refund_stage,
)


@pytest.fixture(autouse=True)
def _clear_idempotency_cache_between_tests():
    """避免 P1-3 引入的进程内幂等缓存跨测试命中。"""
    saved_handlers = dict(tool_registry._handlers)
    saved_capabilities = {cap.name: cap for cap in tool_registry.capabilities()}
    tool_registry._idempotency_cache.clear()
    yield
    tool_registry._capabilities.clear()
    for name, cap in saved_capabilities.items():
        tool_registry._capabilities[name] = cap
    tool_registry._handlers.clear()
    tool_registry._handlers.update(saved_handlers)
    tool_registry._idempotency_cache.clear()


def _make_state(**overrides: Any) -> dict[str, Any]:
    """构造一个最小 AgentState 用于路由测试。"""
    state = default_working_memory()
    state.update(overrides)
    return state


# ============================================================
# A. route_refund_stage 阶段路由 6 阶段覆盖
# ============================================================

def test_route_returns_identify_order_when_no_active_order_id():
    """缺 active_order_id → 阶段 1（IDENTIFY_ORDER）。"""
    state = _make_state()
    assert route_refund_stage(state) == RefundStage.IDENTIFY_ORDER.value


def test_route_returns_collect_reason_when_order_known_but_no_reason():
    """订单已识别但缺退款原因 → 阶段 2（COLLECT_REASON）。"""
    state = _make_state(active_order_id=1, active_order_sn="SN001")
    assert route_refund_stage(state) == RefundStage.COLLECT_REASON.value


def test_route_returns_eligibility_checked_when_reason_known_but_not_checked():
    """订单与原因都有但缺资格校验 → 阶段 3（ELIGIBILITY_CHECKED）。"""
    state = _make_state(
        active_order_id=1,
        active_order_sn="SN001",
        collected_slots={"refund_reason": "尺码不合适"},
    )
    assert route_refund_stage(state) == RefundStage.ELIGIBILITY_CHECKED.value


def test_route_returns_waiting_confirmation_when_eligibility_passed_but_not_confirmed():
    """资格通过但用户未确认 → 阶段 4（WAITING_CONFIRMATION）。"""
    state = _make_state(
        active_order_id=1,
        active_order_sn="SN001",
        collected_slots={"refund_reason": "尺码不合适"},
        last_tool_result={"eligibility_checked": True, "eligibility_passed": True},
    )
    assert route_refund_stage(state) == RefundStage.WAITING_CONFIRMATION.value


def test_route_returns_rejected_when_eligibility_failed():
    state = _make_state(
        active_order_id=1,
        active_order_sn="SN001",
        collected_slots={"refund_reason": "食品变质"},
        last_tool_result={"eligibility_checked": True, "eligibility_passed": False},
    )
    assert route_refund_stage(state) == RefundStage.REJECTED.value


def test_route_returns_submitted_when_all_conditions_met():
    """订单、原因、资格、用户确认都齐备 → 阶段 5（SUBMITTED）。"""
    state = _make_state(
        active_order_id=1,
        active_order_sn="SN001",
        collected_slots={"refund_reason": "尺码不合适", "user_confirmed": True},
        last_tool_result={"eligibility_checked": True, "eligibility_passed": True},
    )
    assert route_refund_stage(state) == RefundStage.SUBMITTED.value


def test_route_returns_done_when_refund_submitted_id_present():
    """已短路（refund_submitted_id 写入） → DONE，跳过所有阶段。"""
    state = _make_state(
        active_order_id=1,
        active_order_sn="SN001",
        collected_slots={
            "refund_reason": "尺码不合适",
            "user_confirmed": True,
            "refund_submitted_id": 42,
        },
        last_tool_result={"eligibility_checked": True, "eligibility_passed": True},
    )
    assert route_refund_stage(state) == RefundStage.DONE.value


# ============================================================
# B. prepare_turn / persist_turn user_confirmed 双向同步
# ============================================================

@pytest.mark.asyncio
async def test_user_confirmed_promotes_from_collected_slots_to_top_level():
    """P1: prepare_turn 必须把 collected_slots.user_confirmed 提升到顶层 user_confirmed，
    让 submit_refund_application 工具通过 InjectedState("user_confirmed") 读到。
    """
    memory = default_working_memory()
    memory["collected_slots"] = {"user_confirmed": True}
    manager = ConversationStateManager(
        extractor=__import__("tests_stubs", fromlist=["StubStructuredLLM"]).StubStructuredLLM()
        if False else _StubExtractor(),
    )
    # 准备一个虚拟 session 对象，让 prepare_turn 跑过同步逻辑
    from types import SimpleNamespace

    session = SimpleNamespace(
        working_memory_json=memory,
        conversation_summary=None,
        version=1,
    )
    # 直接调用 prepare_turn（async）
    result = await manager.prepare_turn(session, "确认提交")
    # 顶层 user_confirmed 必须为 True
    assert result.get("user_confirmed") is True


class _StubExtractor:
    """最小化的 SlotExtraction 桩件，避免 LLM 调用。"""

    async def ainvoke(self, messages):
        from app.conversation.state_manager import SlotExtraction
        return SlotExtraction(
            domain="REFUND",
            explicit_new_goal=True,
            refund_reason=None,
        )


# ============================================================
# C. P1-4 验收 2：未完成确认不能提交
# ============================================================

@pytest.mark.asyncio
async def test_submit_refund_application_rejects_when_user_not_confirmed():
    """submit_refund_application 工具必须在校验 user_confirmed 缺失时直接拒绝。

    P2 起，工具内部走 GuardedToolExecutor → ToolCapabilityRegistry.invoke，
    因此 state 必须显式声明 `workflow_stage=WAITING_CONFIRMATION`（与 P1 流程
    一致）让 stage 检查通过；user_confirmed=False 才会触发 NOT_CONFIRMED 拒绝。
    """
    from typing import Annotated, TypedDict

    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    from app.graph.tools import submit_refund_application

    class S(TypedDict):
        messages: Annotated[list, add_messages]
        user_id: int
        active_order_id: int | None  # P2: Guard required_slot
        active_order_sn: str | None
        collected_slots: dict
        refund_reason_category: str | None
        thread_id: str
        user_confirmed: bool | None
        workflow_stage: str | None
        active_domain: str | None
        last_tool_result: dict | None  # P2: Guard required_eligibility 检查

    workflow = StateGraph(S)
    workflow.add_node("tools", ToolNode([submit_refund_application]))
    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)
    graph = workflow.compile()

    result = await graph.ainvoke(
        {
            "messages": [AIMessage(content="", tool_calls=[{
                "name": "submit_refund_application",
                "args": {},
                "id": "call-no-confirm",
            }])],
            "user_id": 1,
            "thread_id": "t",
            "active_order_id": 1,  # P2: Guard required_slot
            "active_order_sn": "SN20240003",
            "collected_slots": {"refund_reason": "尺码不合适"},
            "refund_reason_category": "SIZE_NOT_FIT",
            "user_confirmed": False,  # 关键：未确认
            "workflow_stage": "WAITING_CONFIRMATION",  # P2: 让 Guard stage check 通过
            "active_domain": "REFUND",  # P2: 让 Guard domain check 通过
            "last_tool_result": {"eligibility_passed": True, "eligibility_checked": True},  # P2: 通过资格校验
        },
        config={"configurable": {"thread_id": "test"}},
    )
    content = result["messages"][-1].content
    assert "❌" in content or "尚未" in content
    assert "尚未确认" in content or "确认" in content


# ============================================================
# D. P1-3 验收 3：已提交申请不会再次创建（DB 短路）
# ============================================================

@pytest.mark.asyncio
async def test_node_submit_short_circuits_when_existing_refund(monkeypatch):
    """node_submit 必须在 order_id 已存在 RefundApplication 时短路返回，
    不调用 submit_refund_application 工具。
    """
    from app.graph.workflows import refund as refund_module

    # 模拟一个已存在的 RefundApplication
    class FakeExisting:
        id = 99
        status = "PENDING"

    async def fake_check_existing(order_id, session):
        return FakeExisting()

    monkeypatch.setattr(
        refund_module.RefundEligibilityChecker,
        "_check_existing_refund",
        fake_check_existing,
    )
    # P2: 拦截 core_submit_refund_application（node_submit 经 GuardedToolExecutor
    # 调用 core handler）。若被调用则测试失败。
    submit_calls: list[dict] = []

    async def fake_core(*, state, idempotency_key=None, **kwargs):
        submit_calls.append({"state": state, "kwargs": kwargs})
        from app.graph.tool_registry import ToolCode, ToolOutcome
        return ToolOutcome(False, ToolCode.SYSTEM_ERROR, "❌ 不应被调用")

    monkeypatch.setattr(refund_module, "core_submit_refund_application", fake_core)
    state = _make_state(
        active_order_id=1,
        active_order_sn="SN001",
        user_id=1,
        collected_slots={"refund_reason": "尺码不合适"},
        last_tool_result={"eligibility_checked": True, "eligibility_passed": True},
        # P1: 顶层 user_confirmed 是工具签名入口；prepare_turn 已从
        # collected_slots.user_confirmed 提升到顶层。
        user_confirmed=True,
    )
    result = await refund_module.node_submit(state)

    # core handler 一次都没被调用（短路分支不调 _executor）
    assert submit_calls == []
    # 已写入短路标志
    assert result["collected_slots"]["refund_submitted_id"] == 99
    assert result["last_tool_result"]["short_circuited"] is True
    assert result["workflow_stage"] == RefundStage.DONE.value
    assert "已有退款申请" in result["answer"]


@pytest.mark.asyncio
async def test_node_submit_refuses_when_user_not_confirmed(monkeypatch):
    """P1-4 验收 2: node_submit 必须在 user_confirmed 缺失时拒绝调工具。"""
    from app.graph.workflows import refund as refund_module

    submit_calls: list[dict] = []

    async def fake_core(*, state, idempotency_key=None, **kwargs):
        from app.graph.tool_registry import ToolCode, ToolOutcome
        submit_calls.append({"state": state, "kwargs": kwargs})
        return ToolOutcome(False, ToolCode.SYSTEM_ERROR, "❌ 不应被调用")

    monkeypatch.setattr(refund_module, "core_submit_refund_application", fake_core)
    state = _make_state(
        active_order_id=1,
        active_order_sn="SN001",
        user_id=1,
        collected_slots={"refund_reason": "尺码不合适"},
        last_tool_result={"eligibility_checked": True, "eligibility_passed": True},
        # user_confirmed 缺失
    )
    result = await refund_module.node_submit(state)

    assert submit_calls == []
    assert "尚未在前端确认" in result["answer"]
    assert result["workflow_stage"] == RefundStage.WAITING_CONFIRMATION.value


# ============================================================
# E. 验收 1：同一槽位最多主动询问一次
# ============================================================

@pytest.mark.asyncio
async def test_collect_reason_node_passes_through_when_already_collected(monkeypatch):
    """P1 验收 1: 槽位已收集后，节点必须直接透传（不再追问）。

    node_collect_reason 在 collected_slots.refund_reason 已存在时返回空 dict，
    不调 LLM、不生成追问话术。这是 FSM 节点纯函数行为，不依赖 DB。
    """
    from app.graph import nodes as graph_nodes
    from app.graph.workflows import refund as refund_module

    class FakeLLM:
        async def astream(self, messages):
            raise AssertionError("❌ 槽位已收集时不应调用 LLM")

    monkeypatch.setattr(graph_nodes, "llm", FakeLLM())

    state = _make_state(
        active_order_id=1,
        active_order_sn="SN001",
        collected_slots={"refund_reason": "尺码不合适"},
        question="我刚才说了原因",
    )
    result = await refund_module.node_collect_reason(state)

    # 透传：返回空 dict，不生成追问话术
    assert result == {}


@pytest.mark.asyncio
async def test_identify_order_node_passes_through_when_order_already_identified(monkeypatch):
    """P1 验收 1（订单号）: active_order_id 已写入时，节点直接透传。"""
    from app.graph import nodes as graph_nodes
    from app.graph.workflows import refund as refund_module

    class FakeLLM:
        async def astream(self, messages):
            raise AssertionError("❌ active_order_id 已存在时不应调用 LLM")

    monkeypatch.setattr(graph_nodes, "llm", FakeLLM())

    state = _make_state(
        active_order_id=42,
        active_order_sn="SN20240003",
        question="我要退款",
    )
    result = await refund_module.node_identify_order(state)

    # 透传：返回空 dict
    assert result == {}


@pytest.mark.asyncio
async def test_prepare_turn_resets_refund_context_when_order_changes():
    memory = default_working_memory()
    memory.update({
        "active_domain": "REFUND",
        "active_order_id": 1,
        "active_order_sn": "SN001",
        "user_confirmed": True,
        "refund_reason_category": "SIZE_NOT_FIT",
        "last_tool_result": {"eligibility_checked": True, "eligibility_passed": True},
        "collected_slots": {
            "order_sn": "SN001", "refund_reason": "尺码不合适",
            "refund_submitted_id": 42, "user_confirmed": True,
        },
    })
    manager = ConversationStateManager(extractor=_StubExtractor())
    from types import SimpleNamespace
    session = SimpleNamespace(working_memory_json=memory, conversation_summary=None, version=1)
    result = await manager.prepare_turn(session, "我要退款，订单号 SN002")
    assert result["active_order_id"] is None
    assert result["user_confirmed"] is None
    assert result["refund_reason_category"] is None
    assert result["last_tool_result"] is None
    assert "refund_submitted_id" not in result["collected_slots"]
