# app/graph/workflows/refund.py
"""P1: RefundWorkflow 6阶段显式 FSM 子图。

阶段流转（与 docs/architecture-update.md 第 5.2 节一致）：
  IDLE
    → IDENTIFY_ORDER          复用 active_order_id；缺失时只问订单号
    → COLLECT_REASON           订单已确认但原因缺失时只问原因
    → ELIGIBILITY_CHECKED      调 check_refund_eligibility，写入 last_tool_result
    → WAITING_CONFIRMATION     展示订单/金额/原因，等前端确认按钮
    → SUBMITTED                调 submit_refund_application 创建申请
    → DONE                     完成（成功提交 / DB 已有申请短路）

P2 changes:
  * 删除 `_eligibility_handler` / `_submit_handler` 中的 `re.search(...)` 和
    `text.startswith("✅")` 字符串解析；改为直接消费 core handler 返回的
    `ToolOutcome`（`outcome.code` / `outcome.data["refund_id"]`）。
  * 所有工具调用统一通过 `GuardedToolExecutor.invoke(...)`，让 Guard 的
    `INVALID_STAGE / MISSING_SLOT / BUSINESS_REJECTED / NOT_CONFIRMED` 校验
    在 RefundWorkflow 与顶层 ToolNode 两条路径上行为一致。
  * Registry 登记 4 个能力：`check_refund_eligibility` / `submit_refund_application`
    / `query_refund_status` / `query_order`（核心订单查询工具）。
  * `last_tool_result` 补齐元数据（tool_name / workflow_stage /
    conversation_id / thread_id / user_id / order_id / timestamp），由 Guard
    自动填充，FSM 不再自行拼装。
"""
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.core.database import async_session_maker
from app.graph.state import AgentState
from app.graph.tool_registry import (
    GuardedToolExecutor,
    ToolCapability,
    ToolCode,
    ToolOutcome,
    tool_registry,
    write_tool_audit_log,
)
from app.graph.tools import (
    core_check_refund_eligibility,
    core_query_refund_status,
    core_submit_refund_application,
)
from app.services.refund_service import RefundEligibilityChecker


class RefundStage(str, Enum):
    IDLE = "IDLE"
    IDENTIFY_ORDER = "IDENTIFY_ORDER"
    COLLECT_REASON = "COLLECT_REASON"
    ELIGIBILITY_CHECKED = "ELIGIBILITY_CHECKED"
    WAITING_CONFIRMATION = "WAITING_CONFIRMATION"
    SUBMITTED = "SUBMITTED"
    DONE = "DONE"
    REJECTED = "REJECTED"


# ===========================================
# 共享执行器（与 tools.py 顶层 LangChain 包装共用同一实例类型）
# ===========================================
_executor = GuardedToolExecutor()


# ===========================================
# 工具函数
# ===========================================

def _slots(state: AgentState) -> dict[str, Any]:
    """安全读取 collected_slots，避免 NotRequired 字段缺失时 NoneType 错误。"""
    slots = state.get("collected_slots")
    return dict(slots) if isinstance(slots, dict) else {}


def _last_result(state: AgentState) -> dict[str, Any]:
    last = state.get("last_tool_result")
    return dict(last) if isinstance(last, dict) else {}


def _stage_payload(stage: RefundStage, **extra: Any) -> dict[str, Any]:
    """统一封装 stage 字段写入。"""
    return {"workflow_stage": stage.value, **extra}


# ===========================================
# 阶段路由（纯函数）
# ===========================================

def route_refund_stage(state: AgentState) -> str:
    """根据工作记忆决定下一阶段（不读 DB）。

    优先级：
      1. collected_slots.refund_submitted_id 存在 → DONE（已提交）
      2. active_order_id 缺失 → IDENTIFY_ORDER
      3. collected_slots.refund_reason 缺失 → COLLECT_REASON
      4. last_tool_result.eligibility_checked 未设置 → ELIGIBILITY_CHECKED
      5. collected_slots.user_confirmed 缺失 → WAITING_CONFIRMATION
      6. 以上都满足 → SUBMITTED
    """
    slots = _slots(state)
    if slots.get("refund_submitted_id"):
        return RefundStage.DONE.value
    if not state.get("active_order_id"):
        return RefundStage.IDENTIFY_ORDER.value
    if not slots.get("refund_reason"):
        return RefundStage.COLLECT_REASON.value
    last = _last_result(state)
    if not last.get("eligibility_checked"):
        return RefundStage.ELIGIBILITY_CHECKED.value
    if last.get("eligibility_passed") is False:
        return RefundStage.REJECTED.value
    if not slots.get("user_confirmed"):
        return RefundStage.WAITING_CONFIRMATION.value
    return RefundStage.SUBMITTED.value


# ===========================================
# 节点函数
# ===========================================

async def node_identify_order(state: AgentState) -> dict[str, Any]:
    """阶段 1: 询问订单号。

    若 working memory 已记录 active_order_id（prepare_turn 通过反查 Order
    表写入），则透传；否则用 LLM 生成追问订单号的话术。
    """
    if state.get("active_order_id"):
        # 已经在 prepare_turn 反查出订单 ID，阶段透传
        return {}
    from app.graph.nodes import llm
    response = None
    async for chunk in llm.astream([
        SystemMessage(content=(
            "你是电商售后助手。当前阶段：询问订单号。"
            "已知订单号时不要重复询问；未知时礼貌请用户提供订单号（如 SN20240001）。"
            "不要调用任何工具，不要输出订单详情。"
        )),
        HumanMessage(content=state["question"]),
    ]):
        response = chunk if response is None else response + chunk
    answer = response.content if response else "请提供要退货的订单号。"
    return {"answer": answer, "messages": [response]}


async def node_collect_reason(state: AgentState) -> dict[str, Any]:
    """阶段 2: 收集退款原因。

    若 working memory 已记录 refund_reason，则透传；否则用 LLM 生成追问原因的话术。
    """
    slots = _slots(state)
    if slots.get("refund_reason"):
        return {}
    from app.graph.nodes import llm
    order_sn = state.get("active_order_sn") or "（未确认）"
    response = None
    async for chunk in llm.astream([
        SystemMessage(content=(
            f"你是电商售后助手。当前阶段：收集退款原因。订单号已确认 = {order_sn}。"
            "请礼貌请用户说明退货原因（自由文本，例如'尺码不合适''商品质量有问题'）。"
            "若用户已给出原因，请直接确认收到，不要重复追问。"
            "不要调用任何工具。"
        )),
        HumanMessage(content=state["question"]),
    ]):
        response = chunk if response is None else response + chunk
    answer = response.content if response else "请说明退货原因。"
    return {"answer": answer, "messages": [response]}


async def node_check_eligibility(state: AgentState) -> dict[str, Any]:
    """阶段 3: 通过 GuardedToolExecutor 调资格检查工具。

    outcome.code 直接决定 FSM 走向：ELIGIBILITY_PASSED → WAITING_CONFIRMATION，
    ELIGIBILITY_REJECTED → REJECTED。FSM 不再依赖字符串前缀判断。
    """
    outcome = await _executor.invoke(
        "check_refund_eligibility",
        state=_workflow_state_dict(state),
    )
    return await _outcome_to_state_update(
        state,
        outcome,
        eligibility_only=True,
        fallback_stage=RefundStage.WAITING_CONFIRMATION
        if outcome.ok
        else RefundStage.REJECTED,
    )


async def node_await_confirmation(state: AgentState) -> dict[str, Any]:
    """阶段 4: 生成确认话术（不调任何工具）。

    话术包含：订单号、退款原因、资格结果；结尾提示用户在前端点击「确认提交」按钮。
    """
    from app.graph.nodes import llm
    slots = _slots(state)
    last = _last_result(state)
    order_sn = state.get("active_order_sn") or "未知"
    reason = slots.get("refund_reason") or "（未提供）"
    eligibility = last.get("eligibility_message") or "（资格未检查）"
    response = None
    async for chunk in llm.astream([
        SystemMessage(content=(
            "你是电商售后助手。当前阶段：等待用户确认退款申请。"
            "请汇总：1) 订单号 2) 退款原因 3) 资格结果；"
            "结尾必须明确提示用户在前端界面点击「确认提交」按钮。"
            "不要自己调用任何工具。"
        )),
        HumanMessage(content=(
            f"订单号：{order_sn}\n退款原因：{reason}\n资格结果：{eligibility}"
        )),
    ]):
        response = chunk if response is None else response + chunk
    answer = response.content if response else (
        f"订单 {order_sn} 退款申请确认中。\n请在前端点击「确认提交」按钮。"
    )
    return {"answer": answer, "messages": [response]}


async def node_submit(state: AgentState) -> dict[str, Any]:
    """阶段 5: 提交退款申请。

    P1-3: 先查 RefundApplication 表，若已有申请则直接短路返回，跳过 submit 工具。
    P1-4: 用户未确认（顶层 user_confirmed=False 或缺失）时拒绝调工具。
    P2: 短路 / 工具调用都走结构化 ToolOutcome，FSM 不再解析"申请编号：#X"。
    """
    order_id = state["active_order_id"]
    slots = _slots(state)

    # P1-4: 用户确认前置校验（与 submit_refund_application 工具内校验一致，
    # 双层防御：子图节点先检查，工具再检查）
    if not state.get("user_confirmed"):
        return {
            **_stage_payload(RefundStage.WAITING_CONFIRMATION),
            "answer": "❌ 尚未在前端确认申请信息。请点击「确认提交」按钮后再发起提交。",
        }

    # P1-3: 查 DB 短路
    async with async_session_maker() as session:
        existing = await RefundEligibilityChecker._check_existing_refund(order_id, session)
    if existing:
        slots = dict(slots)
        slots["refund_submitted_id"] = existing.id
        last = _last_result(state)
        last["short_circuited"] = True
        last["refund_id"] = existing.id
        return {
            **_stage_payload(RefundStage.DONE),
            "collected_slots": slots,
            "last_tool_result": last,
            "answer": (
                f"⏳ 该订单已有退款申请（编号 #{existing.id}，状态 {existing.status}）。\n"
                "无需重复提交，请稍后查询进度。"
            ),
        }

    # 通过 GuardedToolExecutor 调 submit；outcome.code / outcome.data["refund_id"]
    # 是 FSM / 审计 / 管理员界面的唯一真源。
    outcome = await _executor.invoke(
        "submit_refund_application",
        state=_workflow_state_dict(state),
        arguments={
            "reason_detail": slots,
            "reason_category": state.get("refund_reason_category"),
            "user_confirmed": True,
        },
    )
    return await _outcome_to_state_update(
        state,
        outcome,
        eligibility_only=False,
        fallback_stage=RefundStage.DONE
        if outcome.code in {ToolCode.REFUND_SUBMITTED, ToolCode.ALREADY_EXISTS}
        else RefundStage.REJECTED,
    )


async def refund_subgraph_entry(state: AgentState) -> dict[str, Any]:
    """子图入口：根据 route_refund_stage 决定下一节点。

    仅写入 workflow_stage 字段；具体路由由 LangGraph add_conditional_edges 处理。
    """
    next_stage = route_refund_stage(state)
    return _stage_payload(RefundStage(next_stage))


# ===========================================
# 工具登记（启动时一次性注册；测试可在 import 后再补登记）
# ===========================================

def _register_capabilities() -> None:
    if tool_registry.names():
        return
    tool_registry.register(
        ToolCapability(
            name="check_refund_eligibility",
            domain="REFUND",
            allowed_stages=frozenset({RefundStage.ELIGIBILITY_CHECKED.value}),
            required_slots=frozenset({"active_order_sn", "user_id"}),
            writes_to_conversation=True,
            audit_level="read",
            owner_workflow="RefundWorkflow",
        ),
        core_check_refund_eligibility,
    )
    tool_registry.register(
        ToolCapability(
            name="submit_refund_application",
            domain="REFUND",
            allowed_stages=frozenset(
                {RefundStage.SUBMITTED.value, RefundStage.WAITING_CONFIRMATION.value}
            ),
            required_slots=frozenset(
                {"active_order_id", "active_order_sn", "refund_reason", "refund_reason_category"}
            ),
            required_eligibility="passed",
            requires_user_confirmation=True,
            idempotency_key_template="refund:{user_id}:{order_id}",
            writes_to_conversation=True,
            audit_level="sensitive",
            owner_workflow="RefundWorkflow",
        ),
        core_submit_refund_application,
    )
    tool_registry.register(
        ToolCapability(
            name="query_refund_status",
            domain="REFUND",
            allowed_stages=frozenset(
                {
                    RefundStage.IDLE.value,
                    RefundStage.IDENTIFY_ORDER.value,
                    RefundStage.COLLECT_REASON.value,
                    RefundStage.ELIGIBILITY_CHECKED.value,
                    RefundStage.WAITING_CONFIRMATION.value,
                    RefundStage.SUBMITTED.value,
                    RefundStage.DONE.value,
                    RefundStage.REJECTED.value,
                }
            ),
            required_slots=frozenset({"user_id"}),
            writes_to_conversation=True,
            audit_level="read",
            owner_workflow="RefundWorkflow",
        ),
        core_query_refund_status,
    )
    # 核心订单工具（domain=ORDER），在 P2 由独立 core handler 暴露给 Registry；
    # 节点 `query_order` 继续在主图使用，但其底层语义与能力受 Registry 约束。
    from app.graph.tools import core_query_order

    tool_registry.register(
        ToolCapability(
            name="query_order_tool",
            domain="ORDER",
            allowed_stages=frozenset({"*"}),  # 订单查询在所有退款阶段均允许（只读）
            required_slots=frozenset({"user_id"}),
            writes_to_conversation=True,
            audit_level="read",
            owner_workflow="OrderQuery",
        ),
        core_query_order,
    )


# ===========================================
# 工具结果 → 状态写入（结构化）
# ===========================================

async def _outcome_to_state_update(
    state: AgentState,
    outcome: ToolOutcome,
    *,
    eligibility_only: bool,
    fallback_stage: RefundStage,
) -> dict[str, Any]:
    """把 ToolOutcome 转换为节点返回的状态更新。

    不再解析中文 / 正则提取字段——所有结构化字段由 Guard 在 envelope 阶段
    填充（tool_name / workflow_stage / user_id / order_id / timestamp /
    conversation_id / thread_id / idempotency_key），handler 仅消费它们。
    """
    slots = _slots(state)
    last = _last_result(state)

    if eligibility_only:
        last["eligibility_checked"] = True
        last["eligibility_passed"] = outcome.code == ToolCode.ELIGIBILITY_PASSED
        last["eligibility_message"] = outcome.message
        last["tool_outcome"] = _outcome_to_metadata(outcome)
        # 即使 outcome 是 Guard 拒绝（如 INVALID_STAGE），也要把它写回
        # last_tool_result 让后续阶段（如 SUBMITTED）能读取 eligibility_passed。
        last["last_tool_result_code"] = outcome.code
        last["last_tool_result_ok"] = outcome.ok
        return _stage_payload(
            fallback_stage if outcome.ok else RefundStage.REJECTED,
            last_tool_result=last,
            answer=outcome.message,
        )

    # submit 路径
    last["submit_message"] = outcome.message
    last["tool_outcome"] = _outcome_to_metadata(outcome)
    last["last_tool_result_code"] = outcome.code
    last["last_tool_result_ok"] = outcome.ok

    refund_id = (outcome.data or {}).get("refund_id")
    slots = dict(slots)
    if refund_id is not None:
        slots["refund_submitted_id"] = int(refund_id)
        last["refund_id"] = int(refund_id)

    if outcome.code in {ToolCode.REFUND_SUBMITTED, ToolCode.ALREADY_EXISTS}:
        target_stage = RefundStage.DONE
    elif outcome.retryable or outcome.code == ToolCode.TEMPORARY_FAILURE:
        target_stage = RefundStage.SUBMITTED
    else:
        target_stage = fallback_stage

    # 敏感工具：同步写一行审计，避免 fire-and-forget 在测试结束 / 进程
    # 退出前丢失敏感事件记录。审计失败被内部 try/except 吞掉（不会回滚
    # 业务结果），所以 await 是安全的。
    cap = tool_registry.get(outcome.tool_name or "submit_refund_application")
    if cap.audit_level == "sensitive":
        await write_tool_audit_log(outcome)

    return _stage_payload(
        target_stage,
        collected_slots=slots,
        last_tool_result=last,
        answer=outcome.message,
    )


def _outcome_to_metadata(outcome: ToolOutcome) -> dict[str, Any]:
    """把 ToolOutcome 序列化为 last_tool_result.tool_outcome dict。

    包含 PLAN 第 3 节要求的 9+ 项元数据：tool_name, code, ok, message, data,
    retryable, idempotency_key, workflow_stage, conversation_id, thread_id,
    user_id, order_id, timestamp, domain。
    """
    return {
        "tool_name": outcome.tool_name,
        "code": outcome.code,
        "ok": outcome.ok,
        "message": outcome.message,
        "data": outcome.data,
        "retryable": outcome.retryable,
        "idempotency_key": outcome.idempotency_key,
        "workflow_stage": outcome.workflow_stage,
        "conversation_id": outcome.conversation_id,
        "thread_id": outcome.thread_id,
        "user_id": outcome.user_id,
        "order_id": outcome.order_id,
        "timestamp": outcome.timestamp,
        "domain": outcome.domain,
    }


def _workflow_state_dict(state: AgentState) -> dict[str, Any]:
    """把 AgentState 转换为 Registry 需要的 state dict。

    AgentState 的字段都可以直接转 dict；NotRequired 字段缺失时返回 None，
    Guard 内部用 `state.get(key)` 读取时不会 KeyError。
    """
    keys = (
        "user_id", "active_order_id", "active_order_sn", "thread_id",
        "conversation_id", "active_domain", "intent", "workflow_stage",
        "refund_reason_category", "user_confirmed", "collected_slots",
        "last_tool_result",
    )
    return {k: state.get(k) for k in keys}


# ===========================================
# 子图构建
# ===========================================

def build_refund_subgraph():
    """构建并编译 refund 子图（无 checkpointer，由主图托管）。"""
    _register_capabilities()

    workflow = StateGraph(AgentState)

    workflow.add_node("entry", refund_subgraph_entry)
    workflow.add_node("identify_order", node_identify_order)
    workflow.add_node("collect_reason", node_collect_reason)
    workflow.add_node("check_eligibility", node_check_eligibility)
    workflow.add_node("await_confirmation", node_await_confirmation)
    workflow.add_node("submit", node_submit)

    workflow.set_entry_point("entry")

    # entry 根据 workflow_stage 路由到对应节点；DONE 直接结束
    def route_from_entry(state: AgentState) -> str:
        stage = state.get("workflow_stage") or RefundStage.IDENTIFY_ORDER.value
        return stage

    workflow.add_conditional_edges(
        "entry", route_from_entry,
        {
            RefundStage.IDENTIFY_ORDER.value: "identify_order",
            RefundStage.COLLECT_REASON.value: "collect_reason",
            RefundStage.ELIGIBILITY_CHECKED.value: "check_eligibility",
            RefundStage.WAITING_CONFIRMATION.value: "await_confirmation",
            RefundStage.SUBMITTED.value: "submit",
            RefundStage.DONE.value: END,
            RefundStage.REJECTED.value: END,
        },
    )

    # 每轮最多执行一个阶段；下一轮由持久化工作记忆重新路由。
    workflow.add_edge("identify_order", END)
    workflow.add_edge("collect_reason", END)
    workflow.add_edge("check_eligibility", END)
    workflow.add_edge("await_confirmation", END)
    workflow.add_edge("submit", END)  # submit 完成即结束（含短路 / 工具调用两条路径）

    return workflow.compile()


# ===========================================
# Backwards-compat shims for existing imports
# ===========================================
# P1 中的旧符号仍被部分测试 / 旧代码引用；保留以避免破坏外部依赖。
async def _eligibility_handler(*, state: dict[str, Any], idempotency_key: str | None, **_: Any) -> ToolOutcome:
    """[deprecated] P1 兼容入口；P2 起请直接用 `core_check_refund_eligibility`."""
    return await core_check_refund_eligibility(state=state, idempotency_key=idempotency_key)


async def _submit_handler(*, state: dict[str, Any], idempotency_key: str | None, **_: Any) -> ToolOutcome:
    """[deprecated] P1 兼容入口；P2 起请直接用 `core_submit_refund_application`."""
    return await core_submit_refund_application(
        state=state,
        idempotency_key=idempotency_key,
        reason_detail=state.get("collected_slots"),
        reason_category=state.get("refund_reason_category"),
        user_confirmed=True,
    )


def _ensure_tool_capabilities() -> None:
    """[deprecated] P1 兼容入口；P2 起请直接用 `_register_capabilities`."""
    _register_capabilities()