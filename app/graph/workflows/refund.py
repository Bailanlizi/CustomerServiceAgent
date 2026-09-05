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

设计原则（与 architecture-update.md 一致）：
  1. 每个节点只做一件事（推进槽位 / 调工具 / 短路 / 生成话术）。
  2. 阶段路由是纯函数 route_refund_stage(state)，不读 DB。
  3. DB 短路只在 SUBMITTED 节点入口查一次 RefundApplication，写入
     collected_slots.refund_submitted_id，下次路由直接进入 DONE。
  4. 子图节点直接调 refund_tools（手动从 state 提取 InjectedState 字段），
     避免引入 ToolNode 嵌入带来的 free-form 循环。
  5. 话术生成只在 LLM-required 阶段调 LLM（COLLECT_REASON / WAITING_CONFIRMATION），
     其他阶段（IDENTIFY_ORDER / ELIGIBILITY_CHECKED / SUBMITTED / DONE）都是确定性代码。
"""
import re
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.core.database import async_session_maker
from app.graph.state import AgentState
from app.graph.tools import (
    check_refund_eligibility,
    submit_refund_application,
)
from app.services.refund_service import RefundEligibilityChecker
from app.graph.tool_registry import ToolCapability, ToolCode, ToolOutcome, tool_registry


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
    """阶段 3: 调 check_refund_eligibility 工具，写入 last_tool_result。

    工具签名已用 InjectedState 注入 user_id / active_order_sn；这里手动从 state 提取
    InjectedState 字段作为 kwargs 传入，绕过 ToolNode 嵌入。
    """
    state_kwargs = {
        "order_sn": state.get("active_order_sn"),
        "user_id": state.get("user_id"),
    }
    _ensure_tool_capabilities()
    outcome = await tool_registry.invoke("check_refund_eligibility", state, **state_kwargs)
    result = outcome.message
    last = _last_result(state)
    last["eligibility_checked"] = True
    last["eligibility_message"] = result
    # 解析 ✅/❌ 标记以辅助后续路由
    last["eligibility_passed"] = outcome.code == ToolCode.ELIGIBILITY_PASSED
    last["tool_outcome"] = outcome.__dict__
    return _stage_payload(
        RefundStage.ELIGIBILITY_CHECKED,
        last_tool_result=last,
        answer=result,
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
    """
    order_id = state["active_order_id"]
    user_id = state["user_id"]
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

    # 调用 submit_refund_application 工具（手动传参，绕过 ToolNode）
    state_kwargs = {
        "order_sn": state.get("active_order_sn"),
        "user_id": user_id,
        "thread_id": state.get("thread_id"),
        "reason_detail": slots,
        "reason_category": state.get("refund_reason_category"),
        "user_confirmed": True,
    }
    _ensure_tool_capabilities()
    outcome = await tool_registry.invoke("submit_refund_application", state, **state_kwargs)
    result = outcome.message
    slots = dict(slots)
    last = _last_result(state)
    last["submit_message"] = result
    # 解析申请编号 "申请编号：#X"
    refund_id = outcome.data.get("refund_id")
    if refund_id:
        slots["refund_submitted_id"] = int(refund_id)
        last["refund_id"] = int(refund_id)
    last["tool_outcome"] = outcome.__dict__
    if outcome.code in {ToolCode.REFUND_SUBMITTED, ToolCode.ALREADY_EXISTS}:
        outcome_stage = RefundStage.DONE
    elif outcome.retryable:
        outcome_stage = RefundStage.SUBMITTED
    else:
        outcome_stage = RefundStage.REJECTED
    return {
        **_stage_payload(outcome_stage),
        "collected_slots": slots,
        "last_tool_result": last,
        "answer": result,
    }


async def refund_subgraph_entry(state: AgentState) -> dict[str, Any]:
    """子图入口：根据 route_refund_stage 决定下一节点。

    仅写入 workflow_stage 字段；具体路由由 LangGraph add_conditional_edges 处理。
    """
    next_stage = route_refund_stage(state)
    return _stage_payload(RefundStage(next_stage))


# ===========================================
# 子图构建
# ===========================================

def build_refund_subgraph():
    """构建并编译 refund 子图（无 checkpointer，由主图托管）。"""
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


async def _eligibility_handler(*, state: dict[str, Any], idempotency_key: str | None, **_: Any) -> ToolOutcome:
    text = await check_refund_eligibility.ainvoke({"order_sn": state.get("active_order_sn"), "user_id": state.get("user_id")})
    if text.startswith("✅"):
        return ToolOutcome(True, ToolCode.ELIGIBILITY_PASSED, text, {"order_sn": state.get("active_order_sn")})
    return ToolOutcome(False, ToolCode.ELIGIBILITY_REJECTED, text, retryable=False)


async def _submit_handler(*, state: dict[str, Any], idempotency_key: str | None, **_: Any) -> ToolOutcome:
    text = await submit_refund_application.ainvoke({
        "order_sn": state.get("active_order_sn"), "user_id": state.get("user_id"),
        "thread_id": state.get("thread_id"), "reason_detail": state.get("collected_slots"),
        "reason_category": state.get("refund_reason_category"), "user_confirmed": True,
    })
    match = re.search(r"#(\d+)", text)
    if match:
        return ToolOutcome(True, ToolCode.REFUND_SUBMITTED, text, {"refund_id": int(match.group(1))})
    if "已有退款申请" in text or "已存在退款申请" in text:
        match = re.search(r"#(\d+)", text)
        return ToolOutcome(True, ToolCode.ALREADY_EXISTS, text, {"refund_id": int(match.group(1)) if match else None})
    retryable = any(token in text for token in ("稍后重试", "提交失败", "提交冲突", "系统错误"))
    return ToolOutcome(False, ToolCode.TEMPORARY_FAILURE if retryable else ToolCode.BUSINESS_REJECTED, text, retryable=retryable)


def _ensure_tool_capabilities() -> None:
    if tool_registry.names():
        return
    tool_registry.register(ToolCapability(
        name="check_refund_eligibility", domain="REFUND", allowed_stages=frozenset({RefundStage.ELIGIBILITY_CHECKED.value}),
        required_slots=frozenset({"active_order_sn", "user_id"}), writes_to_conversation=True, owner_workflow="RefundWorkflow"), _eligibility_handler)
    tool_registry.register(ToolCapability(
        name="submit_refund_application", domain="REFUND", allowed_stages=frozenset({RefundStage.SUBMITTED.value, RefundStage.WAITING_CONFIRMATION.value}),
        required_slots=frozenset({"active_order_id", "active_order_sn", "refund_reason", "refund_reason_category"}), required_eligibility="passed",
        requires_user_confirmation=True, idempotency_key_template="refund:{user_id}:{order_id}", writes_to_conversation=True, audit_level="sensitive", owner_workflow="RefundWorkflow"), _submit_handler)
