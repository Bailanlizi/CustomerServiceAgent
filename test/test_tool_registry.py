# test/test_tool_registry.py
"""P2 验收测试：ToolCapabilityRegistry + GuardedToolExecutor + ToolOutcome。

覆盖 PLAN 测试计划中 P2 的关键不变量：
  - Registry 注册校验（重复 / 名称唯一 / 写工具幂等键）
  - Guard 拒绝路径（领域 / 阶段 / 槽位 / 资格 / 确认 / handler 不执行）
  - 元数据自动填充（tool_name / workflow_stage / conversation_id / thread_id /
    user_id / order_id / timestamp / idempotency_key）
  - 顶层 ToolNode 走 Guard（与 RefundWorkflow 共享 Outcome.code）
  - admin.py 字段（refund_amount / risk_level / refund_status / 元数据）
  - query_refund_status 与核心订单工具登记
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.graph.tool_registry import (
    GuardedToolExecutor,
    ToolCapability,
    ToolCapabilityRegistry,
    ToolCode,
    ToolOutcome,
    tool_registry,
)


@pytest.fixture(autouse=True)
def _clear_idempotency_cache_between_tests():
    """确保每个测试的幂等缓存和 Registry 都是干净的——之前测试遗留的缓存可能让
    后续测试命中错误的 outcome，遗留的 fake handler 会污染 handler dispatch。

    关键：必须同时清 `_capabilities` 与 `_handlers`，否则
    `_register_capabilities` 检查 `names()` 非空会提前 return，
    后续测试拿不到 handler。
    """
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
from app.graph.tools import (
    check_refund_eligibility,
    core_query_order,
    core_query_refund_status,
    query_order_tool,
    query_refund_status,
    submit_refund_application,
)
from app.graph.workflows import refund as refund_module

# ============================================================
# Fixtures & helpers
# ============================================================

@dataclass
class _CallCounter:
    """记录 handler 调用次数的探针。"""
    calls: int = 0


@pytest.fixture(autouse=True)
def _reset_registry():
    """每个测试结束后清空全局注册表，避免污染后续测试。"""
    yield
    # 不要真正清空 —— _register_capabilities 已经做了"如果已经注册则跳过"。
    # 各测试在隔离状态下创建自己的本地 registry，不影响全局共享实例。


def _fresh_registry() -> ToolCapabilityRegistry:
    """返回一个干净的本地注册表，避免全局状态的污染。"""
    return ToolCapabilityRegistry()


def _ok_handler(outcome: ToolOutcome) -> callable:
    """返回一个把 outcome 当作结果返回的 handler。"""
    async def handler(*, state: dict, idempotency_key: str | None = None, **_: object) -> ToolOutcome:
        return outcome

    handler.__name__ = f"handler_{id(handler)}"
    return handler


# ============================================================
# A. Registry 注册校验
# ============================================================

def test_register_rejects_duplicate_name():
    reg = _fresh_registry()
    cap = ToolCapability(
        name="t", domain="REFUND",
        allowed_stages=frozenset({"DONE"}),
    )
    reg.register(cap, _ok_handler(ToolOutcome(True, ToolCode.SUCCESS, "")))
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(cap, _ok_handler(ToolOutcome(True, ToolCode.SUCCESS, "")))


def test_register_rejects_missing_allowed_stages():
    reg = _fresh_registry()
    cap = ToolCapability(name="t", domain="REFUND", allowed_stages=frozenset())
    with pytest.raises(ValueError, match="allowed_stages"):
        reg.register(cap, _ok_handler(ToolOutcome(True, ToolCode.SUCCESS, "")))


def test_register_rejects_idempotency_key_on_read_tool():
    """PLAN: 只读工具不能配置幂等键。"""
    reg = _fresh_registry()
    cap = ToolCapability(
        name="t", domain="REFUND", allowed_stages=frozenset({"X"}),
        idempotency_key_template="k:{user_id}",
        # writes_to_conversation 默认 False → 触发校验
    )
    with pytest.raises(ValueError, match="idempotency"):
        reg.register(cap, _ok_handler(ToolOutcome(True, ToolCode.SUCCESS, "")))


def test_register_rejects_sensitive_audit_on_read_tool():
    """PLAN: 敏感审计必须是写工具。"""
    reg = _fresh_registry()
    cap = ToolCapability(
        name="t", domain="REFUND", allowed_stages=frozenset({"X"}),
        audit_level="sensitive",
    )
    with pytest.raises(ValueError, match="sensitive"):
        reg.register(cap, _ok_handler(ToolOutcome(True, ToolCode.SUCCESS, "")))


def test_register_rejects_non_callable_handler():
    reg = _fresh_registry()
    cap = ToolCapability(name="t", domain="REFUND", allowed_stages=frozenset({"X"}))
    with pytest.raises(TypeError):
        reg.register(cap, "not a handler")


# ============================================================
# B. Guard 拒绝路径
# ============================================================

@pytest.mark.asyncio
async def test_invoke_rejects_unregistered_tool():
    reg = _fresh_registry()
    with pytest.raises(ValueError, match="unregistered"):
        await reg.invoke("ghost_tool", state={})


@pytest.mark.asyncio
async def test_invoke_rejects_wrong_domain():
    reg = _fresh_registry()
    reg.register(
        ToolCapability(name="t", domain="REFUND", allowed_stages=frozenset({"*"})),
        _ok_handler(ToolOutcome(True, ToolCode.SUCCESS, "")),
    )
    out = await reg.invoke("t", state={"active_domain": "ORDER"})
    assert out.ok is False
    assert out.code == ToolCode.INVALID_STAGE


@pytest.mark.asyncio
async def test_invoke_rejects_wrong_stage():
    reg = _fresh_registry()
    reg.register(
        ToolCapability(name="t", domain="REFUND", allowed_stages=frozenset({"SUBMITTED"})),
        _ok_handler(ToolOutcome(True, ToolCode.SUCCESS, "")),
    )
    out = await reg.invoke("t", state={"active_domain": "REFUND", "workflow_stage": "IDLE"})
    assert out.code == ToolCode.INVALID_STAGE


@pytest.mark.asyncio
async def test_invoke_rejects_missing_required_slot():
    reg = _fresh_registry()
    reg.register(
        ToolCapability(
            name="t", domain="REFUND",
            allowed_stages=frozenset({"X"}),
            required_slots=frozenset({"active_order_sn", "refund_reason"}),
        ),
        _ok_handler(ToolOutcome(True, ToolCode.SUCCESS, "")),
    )
    out = await reg.invoke(
        "t",
        state={"active_domain": "REFUND", "workflow_stage": "X", "active_order_sn": "SN1"},
    )
    assert out.code == ToolCode.MISSING_SLOT
    assert "refund_reason" in out.message


@pytest.mark.asyncio
async def test_invoke_rejects_when_eligibility_not_passed():
    reg = _fresh_registry()
    reg.register(
        ToolCapability(
            name="t", domain="REFUND",
            allowed_stages=frozenset({"X"}),
            required_eligibility="passed",
        ),
        _ok_handler(ToolOutcome(True, ToolCode.SUCCESS, "")),
    )
    out = await reg.invoke(
        "t",
        state={"active_domain": "REFUND", "workflow_stage": "X"},
    )
    assert out.code == ToolCode.BUSINESS_REJECTED


@pytest.mark.asyncio
async def test_invoke_rejects_when_user_not_confirmed():
    reg = _fresh_registry()
    reg.register(
        ToolCapability(
            name="t", domain="REFUND",
            allowed_stages=frozenset({"X"}),
            requires_user_confirmation=True,
        ),
        _ok_handler(ToolOutcome(True, ToolCode.SUCCESS, "")),
    )
    out = await reg.invoke(
        "t",
        state={"active_domain": "REFUND", "workflow_stage": "X", "user_confirmed": False},
    )
    assert out.code == ToolCode.NOT_CONFIRMED


@pytest.mark.asyncio
async def test_guard_does_not_invoke_handler_when_rejected():
    """核心不变量：Guard 拒绝时 handler 一定不被调用。"""
    reg = _fresh_registry()
    counter = _CallCounter()

    async def probe_handler(*, state: dict, idempotency_key: str | None = None, **_: object) -> ToolOutcome:
        counter.calls += 1
        return ToolOutcome(True, ToolCode.SUCCESS, "")

    reg.register(
        ToolCapability(
            name="probe", domain="REFUND",
            allowed_stages=frozenset({"DONE"}),
            requires_user_confirmation=True,
        ),
        probe_handler,
    )
    out = await reg.invoke(
        "probe",
        state={"active_domain": "REFUND", "workflow_stage": "IDLE"},  # 阶段不对
    )
    assert out.code == ToolCode.INVALID_STAGE
    assert counter.calls == 0  # handler 一次都没被调用


# ============================================================
# C. 元数据自动填充
# ============================================================

@pytest.mark.asyncio
async def test_envelope_fills_audit_metadata():
    """ToolOutcome 必须自动填齐 tool_name / workflow_stage / conversation_id /
    thread_id / user_id / order_id / timestamp / idempotency_key / domain."""
    reg = _fresh_registry()
    reg.register(
        ToolCapability(
            name="t", domain="REFUND",
            allowed_stages=frozenset({"X"}),
            idempotency_key_template="k:{user_id}:{order_id}",
            writes_to_conversation=True,
        ),
        _ok_handler(ToolOutcome(True, ToolCode.SUCCESS, "msg", data={"refund_id": 7})),
    )
    out = await reg.invoke(
        "t",
        state={
            "active_domain": "REFUND", "workflow_stage": "X",
            "user_id": 1, "active_order_id": 42,
            "thread_id": "thr-1", "conversation_id": "conv-1",
        },
    )
    assert out.ok is True
    assert out.tool_name == "t"
    assert out.domain == "REFUND"
    assert out.workflow_stage == "X"
    assert out.conversation_id == "conv-1"
    assert out.thread_id == "thr-1"
    assert out.user_id == 1
    assert out.order_id == 42
    assert out.timestamp is not None
    assert out.idempotency_key == "k:1:42"
    assert out.data == {"refund_id": 7}


@pytest.mark.asyncio
async def test_handler_exception_is_enveloped_as_system_error():
    reg = _fresh_registry()

    async def boom(*, state, idempotency_key=None, **_):
        raise RuntimeError("kaboom")

    reg.register(
        ToolCapability(name="t", domain="REFUND", allowed_stages=frozenset({"X"})),
        boom,
    )
    out = await reg.invoke(
        "t",
        state={"active_domain": "REFUND", "workflow_stage": "X"},
    )
    assert out.ok is False
    assert out.code == ToolCode.SYSTEM_ERROR
    assert "kaboom" in out.message


# ============================================================
# D. 顶层 ToolNode 走 Guard（与 RefundWorkflow 共享 Outcome.code）
# ============================================================

class _RecordingHandler:
    """记录被调用次数，并返回固定 ToolOutcome。"""

    def __init__(self, outcome: ToolOutcome) -> None:
        self.outcome = outcome
        self.calls: int = 0

    async def __call__(self, *, state: dict, idempotency_key=None, **_) -> ToolOutcome:
        self.calls += 1
        return self.outcome


@pytest.mark.asyncio
async def test_top_level_toolnode_routes_through_guard():
    """PLAN: 顶层 ToolNode 调 submit_refund_application 时 user_confirmed=False
    必须返回 NOT_CONFIRMED（与 RefundWorkflow 同源）。"""
    handler = _RecordingHandler(ToolOutcome(True, ToolCode.REFUND_SUBMITTED, "OK"))

    reg = _fresh_registry()
    reg.register(
        ToolCapability(
            name="submit_refund_application", domain="REFUND",
            allowed_stages=frozenset({"SUBMITTED", "WAITING_CONFIRMATION"}),
            required_slots=frozenset({"active_order_sn", "user_id"}),
            requires_user_confirmation=True,
            writes_to_conversation=True,
            audit_level="sensitive",
        ),
        handler,
    )

    # 用 GuardedToolExecutor 替换 tools.py 内部的 executor，确保顶层 ToolNode
    # 调用的是这个隔离的 registry。
    executor = GuardedToolExecutor(reg)
    from app.graph import tools as tools_module
    original = tools_module._executor
    tools_module._executor = executor
    try:
        # 让工具的 wrapper 也使用新的 executor（替换实例）
        # 由于 _executor 在 tools.py 是模块级符号，monkeypatch 后所有调用走这里。
        class S(TypedDict):
            messages: Annotated[list, add_messages]
            user_id: int
            active_order_id: int | None
            active_order_sn: str | None
            collected_slots: dict
            refund_reason_category: str | None
            thread_id: str
            user_confirmed: bool | None
            workflow_stage: str | None
            active_domain: str | None

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
                    "id": "test",
                }])],
                "user_id": 1,
                "thread_id": "t",
                "active_order_id": 1,
                "active_order_sn": "SN1",
                "collected_slots": {"refund_reason": "尺码不合适"},
                "refund_reason_category": "SIZE_NOT_FIT",
                "user_confirmed": False,
                "workflow_stage": "WAITING_CONFIRMATION",
                "active_domain": "REFUND",
            },
            config={"configurable": {"thread_id": "test"}},
        )
        content = result["messages"][-1].content
        assert "尚未" in content or "确认" in content
        assert handler.calls == 0  # handler 一次都没被调用
    finally:
        tools_module._executor = original


@pytest.mark.asyncio
async def test_workflow_path_and_toolnode_path_share_same_outcome_code(monkeypatch):
    """PLAN: RefundWorkflow 与顶层 ToolNode 必须返回相同 ToolOutcome.code。

    同源（GuardedToolExecutor → ToolCapabilityRegistry）保证两条路径
    共享 Guard 决策；这里通过 monkeypatch 替换 `core_submit_refund_application`
    为返回固定 outcome 的 handler，验证两条路径都返回相同 code。
    """
    fixed_outcome = ToolOutcome(False, ToolCode.NOT_CONFIRMED, "尚未确认")

    async def fake_core(*, state, idempotency_key=None, **_) -> ToolOutcome:
        return fixed_outcome

    monkeypatch.setattr(refund_module, "core_submit_refund_application", fake_core)
    # 重新触发 _register_capabilities（再次 import 不会重复注册已存在的工具，
    # 但我们的 fake 替换在调用时生效）。
    refund_module._register_capabilities()

    executor = GuardedToolExecutor()

    # RefundWorkflow 路径：node_submit 走 _executor.invoke("submit_refund_application", ...)
    state = {
        "active_domain": "REFUND",
        "workflow_stage": "SUBMITTED",
        "active_order_id": 1, "active_order_sn": "SN1",
        "user_id": 1, "thread_id": "t",
        "collected_slots": {"refund_reason": "尺码不合适"},
        "refund_reason_category": "SIZE_NOT_FIT",
        "last_tool_result": {"eligibility_passed": True, "eligibility_checked": True},
        "user_confirmed": False,
    }
    outcome_workflow = await executor.invoke(
        "submit_refund_application", state=state,
    )
    assert outcome_workflow.code == ToolCode.NOT_CONFIRMED

    # 顶层 ToolNode 路径
    class S(TypedDict):
        messages: Annotated[list, add_messages]
        user_id: int
        active_order_id: int | None
        active_order_sn: str | None
        collected_slots: dict
        refund_reason_category: str | None
        thread_id: str
        user_confirmed: bool | None
        workflow_stage: str | None
        active_domain: str | None

    workflow = StateGraph(S)
    workflow.add_node("tools", ToolNode([submit_refund_application]))
    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)
    graph = workflow.compile()
    result = await graph.ainvoke(
        {
            "messages": [AIMessage(content="", tool_calls=[{
                "name": "submit_refund_application", "args": {}, "id": "x",
            }])],
            "user_id": 1, "thread_id": "t",
            "active_order_id": 1, "active_order_sn": "SN1",
            "collected_slots": {"refund_reason": "尺码不合适"},
            "refund_reason_category": "SIZE_NOT_FIT",
            "user_confirmed": False,
            "workflow_stage": "SUBMITTED",
            "active_domain": "REFUND",
        },
        config={"configurable": {"thread_id": "test"}},
    )
    content = result["messages"][-1].content
    assert "尚未" in content or "确认" in content


# ============================================================
# E. 退款查询工具登记（PLAN 2.3）
# ============================================================

def test_query_refund_status_registered():
    refund_module._register_capabilities()
    cap = refund_module.tool_registry.get("query_refund_status")
    assert cap.domain == "REFUND"
    assert "user_id" in cap.required_slots
    assert cap.audit_level == "read"


def test_query_order_registered_with_order_domain():
    refund_module._register_capabilities()
    cap = refund_module.tool_registry.get("query_order_tool")
    assert cap.domain == "ORDER"
    assert "user_id" in cap.required_slots


def test_all_refund_tools_are_in_registry():
    refund_module._register_capabilities()
    for tool in (check_refund_eligibility, submit_refund_application, query_refund_status, query_order_tool):
        name = tool.name
        # LangChain StructuredTool 用 `name` 属性暴露原 @tool name。
        cap = refund_module.tool_registry.get(name)
        assert cap is not None


# ============================================================
# F. core_query_order 行为
# ============================================================

@pytest.mark.asyncio
async def test_core_query_order_returns_success_outcome():
    """core_query_order 应返回结构化 ToolOutcome；FSM / 审计读取 data 字段。"""

    # 用一个不存在的订单号（避免 DB 依赖）：应当返回 NOT_AUTHORIZED。
    out = await core_query_order(state={"user_id": 999, "active_order_sn": None}, question="")
    # 不强制 DB 依赖：要么 SUCCESS（如果 seed 数据存在），要么 NOT_AUTHORIZED。
    assert out.code in {ToolCode.SUCCESS, ToolCode.NOT_AUTHORIZED}
    assert out.tool_name is None  # core handler 不填 tool_name，由 Guard 填


# ============================================================
# G. core_query_refund_status 行为
# ============================================================

@pytest.mark.asyncio
async def test_core_query_refund_status_returns_tool_outcome():
    out = await core_query_refund_status(state={"user_id": 999})
    # 用户 999 无申请 → SUCCESS with data.refunds=[]
    assert out.code == ToolCode.SUCCESS
    assert out.data.get("refunds") == []
    assert "📭" in out.message


# ============================================================
# H. admin.py 元数据字段
# ============================================================

def test_audit_task_includes_p2_fields():
    """admin.py AuditTask 必须包含 P2 新增的元数据字段。"""
    from app.api.v1.admin import AuditTask

    fields = AuditTask.model_fields.keys()
    for f in (
        "refund_amount", "refund_risk_level", "refund_status",
        "workflow_stage", "user_confirmed",
        "last_tool_outcome", "eligibility_result",
    ):
        assert f in fields, f"missing field: {f}"


# ============================================================
# I. _outcome_to_metadata 元数据覆盖
# ============================================================

def test_outcome_to_metadata_includes_plan_required_fields():
    """PLAN 第 3 节要求 9+ 项元数据写入 last_tool_result.tool_outcome。"""
    refund_module._register_capabilities()
    out = ToolOutcome(
        ok=True, code=ToolCode.REFUND_SUBMITTED, message="OK",
        data={"refund_id": 7},
        tool_name="submit_refund_application", domain="REFUND",
        workflow_stage="SUBMITTED", conversation_id="conv-1",
        thread_id="thr-1", user_id=1, order_id=42, timestamp="2026-01-01T00:00:00",
        idempotency_key="refund:1:42",
    )
    meta = refund_module._outcome_to_metadata(out)
    for key in (
        "tool_name", "code", "ok", "message", "data", "retryable",
        "idempotency_key", "workflow_stage", "conversation_id", "thread_id",
        "user_id", "order_id", "timestamp", "domain",
    ):
        assert key in meta, f"missing metadata key: {key}"
    assert meta["refund_id" if False else "data"]["refund_id"] == 7


# ============================================================
# J. P2 收尾验收（修复 codex review 提出的问题）
# ============================================================


def test_wildcard_allowed_stages_match_any_stage():
    """P0-3 修复：allowed_stages=frozenset({"*"}) 必须匹配任意 stage，
    不应让 query_order_tool 在所有实际阶段都被判定为 INVALID_STAGE。
    """
    import asyncio

    reg = ToolCapabilityRegistry()

    async def handler(*, state, idempotency_key, **_):
        return ToolOutcome(ok=True, code=ToolCode.SUCCESS, message="ok")

    reg.register(
        ToolCapability(
            name="wildcard_tool",
            domain="ORDER",
            allowed_stages=frozenset({"*"}),
            required_slots=frozenset({"user_id"}),
            writes_to_conversation=True,
            audit_level="read",
        ),
        handler,
    )

    async def _run_all():
        for stage in (
            "IDLE", "IDENTIFY_ORDER", "WAITING_CONFIRMATION",
            "DONE", "REJECTED", None,
        ):
            out = await reg.invoke(
                "wildcard_tool",
                state={"user_id": 1, "workflow_stage": stage, "active_domain": "ORDER"},
            )
            assert out.ok, f"wildcard stage={stage!r} should pass Guard"

    asyncio.run(_run_all())


def test_wildcard_still_blocks_other_domains():
    """通配只豁免 stage 检查，不豁免 domain 检查。"""
    reg = ToolCapabilityRegistry()

    async def handler(*, state, idempotency_key, **_):
        return ToolOutcome(ok=True, code=ToolCode.SUCCESS, message="ok")

    reg.register(
        ToolCapability(
            name="order_wildcard",
            domain="ORDER",
            allowed_stages=frozenset({"*"}),
            required_slots=frozenset({"user_id"}),
            writes_to_conversation=True,
        ),
        handler,
    )
    import asyncio
    out = asyncio.run(reg.invoke(
        "order_wildcard",
        state={"user_id": 1, "active_domain": "REFUND", "workflow_stage": "WAITING_CONFIRMATION"},
    ))
    assert not out.ok
    assert out.code == ToolCode.INVALID_STAGE


@pytest.mark.asyncio
async def test_audit_log_writes_with_legal_audit_action(monkeypatch):
    """P0-2 修复：write_tool_audit_log 必须使用 AuditAction.REJECT（不是
    REJECTED），否则 AuditAction.REJECTED 抛 AttributeError 被吞掉，敏感
    审计事件全部丢失。
    """
    from app.graph.tool_registry import write_tool_audit_log
    from app.models.audit import AuditAction

    class FakeSession:
        def __init__(self):
            self.added = []
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
        def add(self, obj):
            obj.id = len(self.added) + 1  # 模拟主键生成
            self.added.append(obj)
        async def commit(self):
            return None
        async def refresh(self, obj):
            return None

    fake = FakeSession()

    # 关键：write_tool_audit_log 在 tool_registry 模块内 import 了
    # async_session_maker，monkeypatch 必须在同一模块生效。
    monkeypatch.setattr(
        "app.graph.tool_registry.async_session_maker",
        lambda: fake,
    )

    outcome = ToolOutcome(
        ok=False,
        code=ToolCode.BUSINESS_REJECTED,
        message="资格未通过",
        tool_name="submit_refund_application",
        domain="REFUND",
        workflow_stage="SUBMITTED",
        thread_id="thr-1",
        user_id=1,
        order_id=42,
        timestamp="2026-01-01T00:00:00",
    )

    audit_id = await write_tool_audit_log(outcome)

    assert audit_id is not None
    assert len(fake.added) == 1
    audit_log = fake.added[0]
    # 关键断言：写入的 action 必须是合法枚举值（不是 REJECTED）
    assert audit_log.action in {
        AuditAction.PENDING,
        AuditAction.APPROVE,
        AuditAction.REJECT,
        AuditAction.ESCALATE,
    }
    assert audit_log.action == AuditAction.REJECT  # 失败 outcome 应为 REJECT


@pytest.mark.asyncio
async def test_idempotency_cache_hit_on_repeat_submit(monkeypatch):
    """P1-3 收尾：Registry 层幂等命中，重复 submit 调用复用同一 outcome，
    不再次执行 handler 业务逻辑。
    """
    from app.graph.tool_registry import tool_registry
    from app.graph.workflows import refund as refund_module

    refund_module._register_capabilities()

    calls: list[int] = []

    async def fake_core_submit(*, state, idempotency_key, **_):
        calls.append(1)
        return ToolOutcome(
            ok=True,
            code=ToolCode.REFUND_SUBMITTED,
            message="已提交",
            data={"refund_id": 100},  # 固定 ID，用于验证幂等命中
            idempotency_key=idempotency_key,
        )

    # 关键：Registry 在 _register_capabilities 时把 handler 引用固化到
    # `_handlers[name]`，所以 monkeypatch 模块级符号无效；必须直接替换
    # 已存储的引用。
    tool_registry._handlers["submit_refund_application"] = fake_core_submit

    executor = refund_module._executor
    base_state = {
        "user_id": 1,
        "active_order_id": 999,
        "active_order_sn": "SN1",
        "thread_id": "thr",
        "conversation_id": "conv",
        "workflow_stage": "SUBMITTED",
        "active_domain": "REFUND",
        "refund_reason_category": "QUALITY_ISSUE",
        "user_confirmed": True,
        "collected_slots": {"refund_reason": "尺码不合适"},
        "last_tool_result": {"eligibility_checked": True, "eligibility_passed": True},
    }

    out1 = await executor.invoke(
        "submit_refund_application",
        state=base_state,
        arguments={
            "reason_detail": base_state["collected_slots"],
            "reason_category": "QUALITY_ISSUE",
            "user_confirmed": True,
        },
    )
    assert out1.ok
    assert out1.code == ToolCode.REFUND_SUBMITTED
    assert out1.data["refund_id"] == 100

    # 第二次：handler 不应被调，outcome 命中缓存
    out2 = await executor.invoke(
        "submit_refund_application",
        state=base_state,
        arguments={
            "reason_detail": base_state["collected_slots"],
            "reason_category": "QUALITY_ISSUE",
            "user_confirmed": True,
        },
    )
    assert out2.ok
    # 与第一次完全一致（数据 + code + idempotency_key）
    assert out2.data["refund_id"] == out1.data["refund_id"]
    assert out2.code == out1.code
    assert out2.idempotency_key == out1.idempotency_key
    assert out2.idempotency_key == "refund:1:999"
    # handler 只被调一次（命中缓存后不再调用）
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_idempotency_cache_does_not_cache_failed_outcome(monkeypatch):
    """失败 outcome 不进入缓存——下次相同 key 必须重新尝试。"""
    from app.graph.tool_registry import tool_registry
    from app.graph.workflows import refund as refund_module

    refund_module._register_capabilities()

    calls: list[int] = []

    async def fake_core_submit(*, state, idempotency_key, **_):
        calls.append(1)
        if len(calls) == 1:
            return ToolOutcome(
                ok=False,
                code=ToolCode.TEMPORARY_FAILURE,
                message="数据库暂时不可用",
                retryable=True,
            )
        return ToolOutcome(
            ok=True,
            code=ToolCode.REFUND_SUBMITTED,
            message="重试后成功",
            data={"refund_id": 200},
        )

    tool_registry._handlers["submit_refund_application"] = fake_core_submit
    # 清掉先前测试遗留的幂等缓存，避免测试间相互污染
    tool_registry._idempotency_cache.clear()

    executor = refund_module._executor
    base_state = {
        "user_id": 1,
        "active_order_id": 999,
        "active_order_sn": "SN1",
        "thread_id": "thr",
        "workflow_stage": "SUBMITTED",
        "active_domain": "REFUND",
        "refund_reason_category": "QUALITY_ISSUE",
        "user_confirmed": True,
        "collected_slots": {"refund_reason": "尺码不合适"},
        "last_tool_result": {"eligibility_checked": True, "eligibility_passed": True},
    }

    out1 = await executor.invoke(
        "submit_refund_application",
        state=base_state,
        arguments={"reason_detail": base_state["collected_slots"]},
    )
    assert not out1.ok
    assert out1.retryable

    out2 = await executor.invoke(
        "submit_refund_application",
        state=base_state,
        arguments={"reason_detail": base_state["collected_slots"]},
    )
    assert out2.ok
    assert out2.data["refund_id"] == 200
    assert len(calls) == 2  # 失败不缓存，第二次真的执行了 handler