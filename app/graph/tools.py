# app/graph/tools.py
"""
LangGraph Tools: Agent 可调用的工具函数

P2 architecture:
  * Each tool has a *core handler* returning a structured `ToolOutcome`. The
    handler is the single source of truth for the result.
  * The LangChain `@tool` wrapper is a thin shell that pulls parameters from
    InjectedState, calls the core handler, and returns `outcome.message` as
    a string (LangChain `ToolMessage` contract). All structured fields live on
    `outcome.data` / `outcome.code`, never encoded into the string.
  * Both the LangChain ToolNode path and the RefundWorkflow path funnel
    through `GuardedToolExecutor` → `ToolCapabilityRegistry.invoke`, so Guard
    checks (stage / domain / slot / eligibility / confirmation / idempotency)
    run uniformly.
"""
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from sqlmodel import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.graph.tool_registry import (
    GuardedToolExecutor,
    ToolCode,
    ToolOutcome,
)
from app.models.audit import AuditAction, AuditLog, RiskLevel
from app.models.order import Order
from app.services.refund_service import (
    RefundApplicationService,
    RefundEligibilityChecker,
    RefundReason,
)
from app.tasks.refund_tasks import notify_admin_audit
from app.websocket.manager import manager

# ==========================================================
# 共享执行器
# ==========================================================
# All LangChain-side and workflow-side executions funnel through this
# executor, so the same Guard checks run on every call.
_executor = GuardedToolExecutor()


def _ensure_capabilities_registered() -> None:
    """Lazy registration so LangChain ToolNode 调用前 Registry 必有相应能力。

    Importing `app.graph.workflows.refund` 与 `app.graph.workflows.order` 会分别注册
    REFUND 与 ORDER 两组能力；两者均按 per-name 幂等，避免重复 import 时重复注册。
    """
    from app.graph.workflows import order as _order
    from app.graph.workflows import refund as _refund

    _refund._register_capabilities()
    _order._register_capabilities()


# ==========================================================
# Core handler 1: 退货资格预检
# ==========================================================
async def core_check_refund_eligibility(
    *,
    state: dict,
    idempotency_key: str | None = None,
    **_: object,
) -> ToolOutcome:
    """检查订单是否符合退货条件（core handler）。"""
    order_sn = state.get("active_order_sn")
    user_id = state.get("user_id")
    if not order_sn:
        return ToolOutcome(
            ok=False,
            code=ToolCode.MISSING_SLOT,
            message="❌ 缺少订单号，无法进行资格预检。请先告知要查询的订单号。",
        )
    if not user_id:
        return ToolOutcome(
            ok=False,
            code=ToolCode.NOT_AUTHORIZED,
            message="❌ 缺少用户身份，无法进行资格预检。",
        )

    async with async_session_maker() as session:
        stmt = select(Order).where(
            Order.order_sn == order_sn,
            Order.user_id == user_id,
        )
        order = (await session.exec(stmt)).first()
        if not order:
            return ToolOutcome(
                ok=False,
                code=ToolCode.NOT_AUTHORIZED,
                message=f"❌ 未找到订单 {order_sn}，或您无权访问此订单。",
            )

        is_eligible, message = await RefundEligibilityChecker.check_eligibility(order, session)

        if is_eligible:
            return ToolOutcome(
                ok=True,
                code=ToolCode.ELIGIBILITY_PASSED,
                message=(
                    f"✅ 订单 {order_sn} 符合退货条件。\n"
                    f"订单信息：\n"
                    f"  - 商品：{', '.join([item['name'] for item in order.items])}\n"
                    f"  - 金额：¥{order.total_amount}\n"
                    f"  - 状态：{order.status}\n"
                    f"检查结果：{message}"
                ),
                data={
                    "order_sn": order_sn,
                    "order_id": order.id,
                    "total_amount": str(order.total_amount),
                    "items": order.items,
                    "status": str(order.status),
                },
            )
        return ToolOutcome(
            ok=False,
            code=ToolCode.ELIGIBILITY_REJECTED,
            message=f"❌ 订单 {order_sn} 不符合退货条件。\n拒绝原因：{message}",
            data={"order_sn": order_sn, "order_id": order.id, "reason": message},
        )


@tool
async def check_refund_eligibility(
    order_sn: Annotated[str | None, InjectedState("active_order_sn")] = None,
    user_id: Annotated[int | None, InjectedState("user_id")] = None,
    workflow_stage: Annotated[str | None, InjectedState("workflow_stage")] = None,
    active_domain: Annotated[str | None, InjectedState("active_domain")] = None,
) -> str:
    """检查订单是否符合退货条件。

    使用场景：
    - 用户询问"我的订单能退货吗？"
    - 在正式申请退货前进行资格预检

    前置约束：
    - 订单号必须已写入会话工作记忆（active_order_sn），否则提示用户补全。

    返回：
    - 如果可以退货，返回"符合退货条件"及详细说明
    - 如果不能退货，返回拒绝原因（如：超期、已退、商品类别等）
    """
    # 构造最小 state dict 供 core handler 消费；元数据由 Registry 注入。
    _ensure_capabilities_registered()
    outcome = await _executor.invoke(
        "check_refund_eligibility",
        state={
            "active_order_sn": order_sn,
            "user_id": user_id,
            "workflow_stage": workflow_stage,
            "active_domain": active_domain,
        },
    )
    return outcome.message


# ==========================================================
# Core handler 2: 提交退货申请
# ==========================================================
async def core_submit_refund_application(
    *,
    state: dict,
    idempotency_key: str | None = None,
    reason_detail: dict | None = None,
    reason_category: str | None = None,
    user_confirmed: bool | None = None,
    **_: object,
) -> ToolOutcome:
    """提交退货申请（core handler）。"""
    order_sn = state.get("active_order_sn")
    user_id = state.get("user_id")
    thread_id = state.get("thread_id")

    if not order_sn:
        return ToolOutcome(
            ok=False,
            code=ToolCode.MISSING_SLOT,
            message="❌ 缺少订单号，无法提交退款申请。请先告知要退货的订单号。",
        )
    if not user_id:
        return ToolOutcome(
            ok=False,
            code=ToolCode.NOT_AUTHORIZED,
            message="❌ 缺少用户身份，无法提交退款申请。",
        )
    if not user_confirmed:
        return ToolOutcome(
            ok=False,
            code=ToolCode.NOT_CONFIRMED,
            message=(
                "❌ 尚未确认申请信息。请在前端界面点击「确认提交」按钮，"
                "确认订单号、退款原因与退款金额后再发起提交。"
            ),
        )

    slots = reason_detail if isinstance(reason_detail, dict) else {}
    reason_text = slots.get("refund_reason") if isinstance(slots, dict) else None
    if not reason_text:
        return ToolOutcome(
            ok=False,
            code=ToolCode.MISSING_SLOT,
            message="❌ 缺少退货原因，无法提交退款申请。请先说明退货原因（如质量问题、尺码不合适等）。",
        )

    async with async_session_maker() as session:
        order = (
            await session.exec(
                select(Order).where(
                    Order.order_sn == order_sn,
                    Order.user_id == user_id,
                )
            )
        ).first()
        if not order:
            return ToolOutcome(
                ok=False,
                code=ToolCode.NOT_AUTHORIZED,
                message=f"❌ 未找到订单 {order_sn}，或您无权访问此订单。",
            )

        category = None
        if reason_category:
            try:
                category = RefundReason(reason_category)
            except ValueError:
                category = RefundReason.OTHER

        success, message, refund_app = await RefundApplicationService.create_refund_application(
            order_id=order.id,
            user_id=user_id,
            reason_detail=reason_text,
            reason_category=category,
            session=session,
            commit=False,
        )

        if success and refund_app:
            refund_amount = float(refund_app.refund_amount)
            risk_level = (
                RiskLevel.HIGH
                if refund_amount >= settings.HIGH_RISK_REFUND_AMOUNT
                else RiskLevel.MEDIUM
                if refund_amount >= settings.MEDIUM_RISK_REFUND_AMOUNT
                else RiskLevel.LOW
            )
            threshold = (
                settings.HIGH_RISK_REFUND_AMOUNT
                if risk_level == RiskLevel.HIGH
                else settings.MEDIUM_RISK_REFUND_AMOUNT
                if risk_level == RiskLevel.MEDIUM
                else 0
            )
            trigger_reason = f"{risk_level} 风险退款申请：¥{refund_amount}"
            if threshold:
                trigger_reason += f" (≥ ¥{threshold})"
            audit_log = AuditLog(
                thread_id=thread_id,
                user_id=user_id,
                order_id=order.id,
                refund_application_id=refund_app.id,
                trigger_reason=trigger_reason,
                risk_level=risk_level,
                action=AuditAction.PENDING,
                context_snapshot={
                    "question": reason_text,
                    "order_data": {
                        "order_id": order.id,
                        "order_sn": order.order_sn,
                        "status": str(order.status),
                        "total_amount": str(order.total_amount),
                        "items": order.items,
                    },
                    "refund": {
                        "refund_application_id": refund_app.id,
                        "refund_amount": str(refund_app.refund_amount),
                        "reason": reason_text,
                        "reason_category": category.value if category else None,
                    },
                    "idempotency_key": idempotency_key,
                },
            )
            session.add(audit_log)
            await session.commit()
            await session.refresh(audit_log)

            try:
                notify_admin_audit.delay(audit_log.id)
                await manager.notify_status_change(
                    thread_id=thread_id,
                    status="WAITING_ADMIN",
                    data={
                        "risk_level": risk_level,
                        "audit_log_id": audit_log.id,
                        "refund_amount": refund_amount,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - 通知失败不能回滚已写入的申请和审计。
                print(f"[Refund tool] 审核通知失败: {exc}")

            return ToolOutcome(
                ok=True,
                code=ToolCode.REFUND_SUBMITTED,
                message=(
                    f"⏳ 退货申请已提交，需人工审核。\n"
                    f"申请编号：#{refund_app.id}\n退款金额：¥{refund_amount}\n"
                    f"风险等级：{risk_level}\n触发原因：{trigger_reason}\n"
                    "申请已进入人工审核，请稍后查询进度。"
                ),
                data={
                    "refund_id": int(refund_app.id),
                    "refund_amount": str(refund_app.refund_amount),
                    "risk_level": str(risk_level),
                    "audit_log_id": int(audit_log.id),
                },
            )

        if refund_app:
            return ToolOutcome(
                ok=False,
                code=ToolCode.ALREADY_EXISTS,
                message=f"❌ 该订单已有退款申请。\n{message}",
                data={"refund_id": int(refund_app.id), "status": str(refund_app.status)},
            )

        return ToolOutcome(
            ok=False,
            code=ToolCode.BUSINESS_REJECTED,
            message=f"❌ 退货申请失败。\n原因：{message}",
            retryable=False,
        )


@tool
async def submit_refund_application(
    order_sn: Annotated[str | None, InjectedState("active_order_sn")] = None,
    order_id: Annotated[int | None, InjectedState("active_order_id")] = None,
    user_id: Annotated[int | None, InjectedState("user_id")] = None,
    thread_id: Annotated[str | None, InjectedState("thread_id")] = None,
    workflow_stage: Annotated[str | None, InjectedState("workflow_stage")] = None,
    active_domain: Annotated[str | None, InjectedState("active_domain")] = None,
    last_tool_result: Annotated[dict | None, InjectedState("last_tool_result")] = None,
    reason_detail: Annotated[dict | None, InjectedState("collected_slots")] = None,
    reason_category: Annotated[
        str | None,
        InjectedState("refund_reason_category"),
    ] = None,
    user_confirmed: Annotated[bool | None, InjectedState("user_confirmed")] = None,
) -> str:
    """提交退货申请。

    使用场景：
    - 用户明确表示"我要退货"
    - 工作记忆已记录订单号与退款原因（free-text + enum 分类）

    注意：
    - 订单号与退款原因必须由工作记忆提供，缺失时直接拒绝并提示用户补全，避免
      "前端说订单号已经被记住、工具仍要求再问一遍"的体感反复。
    - 此工具会自动校验退货资格。如果资格不符，会直接拒绝并返回原因。
    - 成功后会生成退货申请记录并触发审核通知。

    返回：
    - 成功：返回申请编号和后续流程说明
    - 失败：返回拒绝原因
    """
    _ensure_capabilities_registered()
    outcome = await _executor.invoke(
        "submit_refund_application",
        state={
            "active_order_sn": order_sn,
            "active_order_id": order_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "workflow_stage": workflow_stage,
            "active_domain": active_domain,
            "user_confirmed": user_confirmed,
            "collected_slots": reason_detail,
            "refund_reason_category": reason_category,
            "last_tool_result": last_tool_result,
        },
        arguments={
            "reason_detail": reason_detail,
            "reason_category": reason_category,
            "user_confirmed": user_confirmed,
        },
    )
    return outcome.message


# ==========================================================
# Core handler 3: 查询退货申请状态
# ==========================================================
async def core_query_refund_status(
    *,
    state: dict,
    idempotency_key: str | None = None,
    refund_id: int | None = None,
    **_: object,
) -> ToolOutcome:
    """查询退货申请状态（core handler）。"""
    user_id = state.get("user_id")
    if not user_id:
        return ToolOutcome(
            ok=False,
            code=ToolCode.NOT_AUTHORIZED,
            message="❌ 缺少用户身份，无法查询退款申请。",
        )

    async with async_session_maker() as session:
        if refund_id:
            refund = await RefundApplicationService.get_refund_by_id(
                refund_id=refund_id,
                user_id=user_id,
                session=session,
            )
            if not refund:
                return ToolOutcome(
                    ok=False,
                    code=ToolCode.NOT_AUTHORIZED,
                    message=f"❌ 未找到申请编号 #{refund_id}，或您无权访问此申请。",
                )
            order = (
                await session.exec(select(Order).where(Order.id == refund.order_id))
            ).first()

            review_info = (
                f"审核信息：\n  - 审核时间：{refund.reviewed_at.strftime('%Y-%m-%d %H:%M')}"
                if refund.reviewed_at
                else "⏳ 审核中，请耐心等待"
            )
            review_note = f"  - 审核备注：{refund.admin_note}" if refund.admin_note else ""

            return ToolOutcome(
                ok=True,
                code=ToolCode.SUCCESS,
                message=(
                    f"📋 退货申请详情（#{refund.id}）\n\n"
                    f"订单信息：\n"
                    f"  - 订单号：{order.order_sn if order else '未知'}\n"
                    f"  - 商品：{', '.join([item['name'] for item in order.items]) if order else '未知'}\n\n"
                    f"申请信息：\n"
                    f"  - 申请状态：{refund.status}\n"
                    f"  - 退款金额：¥{refund.refund_amount}\n"
                    f"  - 申请时间：{refund.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                    f"  - 退货原因：{refund.reason_detail}\n\n"
                    f"{review_info}\n"
                    f"{review_note}"
                ),
                data={
                    "refund_id": int(refund.id),
                    "status": str(refund.status),
                    "refund_amount": str(refund.refund_amount),
                    "order_sn": order.order_sn if order else None,
                },
            )

        refund_list = await RefundApplicationService.get_user_refund_applications(
            user_id=user_id,
            session=session,
        )
        if not refund_list:
            return ToolOutcome(
                ok=True,
                code=ToolCode.SUCCESS,
                message="📭 您还没有退货申请记录。",
                data={"refunds": []},
            )

        result_text = f"📋 您的退货申请列表（共 {len(refund_list)} 条）\n\n"
        refunds_data = []
        for refund in refund_list:
            order = (
                await session.exec(select(Order).where(Order.id == refund.order_id))
            ).first()

            status_emoji = {
                "PENDING": "⏳",
                "APPROVED": "✅",
                "PROCESSING": "💳",
                "REJECTED": "❌",
                "COMPLETED": "🎉",
                "CANCELLED": "🚫",
            }.get(refund.status, "❓")

            result_text += (
                f"{status_emoji} 申请 #{refund.id}\n"
                f"  订单号：{order.order_sn if order else '未知'}\n"
                f"  状态：{refund.status}\n"
                f"  金额：¥{refund.refund_amount}\n"
                f"  申请时间：{refund.created_at.strftime('%Y-%m-%d')}\n\n"
            )
            refunds_data.append(
                {
                    "refund_id": int(refund.id),
                    "status": str(refund.status),
                    "refund_amount": str(refund.refund_amount),
                    "order_sn": order.order_sn if order else None,
                }
            )

        return ToolOutcome(
            ok=True,
            code=ToolCode.SUCCESS,
            message=result_text.strip(),
            data={"refunds": refunds_data},
        )


@tool
async def query_refund_status(
    user_id: Annotated[int | None, InjectedState("user_id")] = None,
    active_domain: Annotated[str | None, InjectedState("active_domain")] = None,
    refund_id: int | None = None,
) -> str:
    """查询退货申请状态。

    使用场景：
    - 用户询问"我的退货申请怎么样了？"
    - 用户提供申请编号查询具体状态
    - 用户想查看所有退货记录

    返回：
    - 如果指定申请编号：返回该申请的详细信息
    - 如果未指定：返回用户所有退货申请列表
    """
    _ensure_capabilities_registered()
    outcome = await _executor.invoke(
        "query_refund_status",
        state={"user_id": user_id, "active_domain": active_domain},
        arguments={"refund_id": refund_id},
    )
    return outcome.message


# ==========================================================
# Core handler 4: 核心订单查询（domain=ORDER）
# ==========================================================
async def core_query_order(
    *,
    state: dict,
    idempotency_key: str | None = None,
    question: str | None = None,
    **_: object,
) -> ToolOutcome:
    """查询订单（core handler，domain=ORDER）。"""
    user_id = state.get("user_id")
    if not user_id:
        return ToolOutcome(
            ok=False,
            code=ToolCode.NOT_AUTHORIZED,
            message="❌ 缺少用户身份，无法查询订单。",
        )

    # 优先使用工作记忆中的 active_order_sn，避免模型伪造订单号。
    order_sn = state.get("active_order_sn")
    text = question or ""

    import re
    if not order_sn:
        match = re.search(r"SN\d+", text.upper())
        if match:
            order_sn = match.group()

    async with async_session_maker() as session:
        if order_sn:
            order = (
                await session.exec(
                    select(Order).where(
                        Order.order_sn == order_sn,
                        Order.user_id == user_id,
                    )
                )
            ).first()
        else:
            # 无显式订单号时回退到最近一个订单。
            order = (
                await session.exec(
                    select(Order)
                    .where(Order.user_id == user_id)
                    .order_by(Order.created_at.desc())
                    .limit(1)
                )
            ).first()

    if not order:
        return ToolOutcome(
            ok=False,
            code=ToolCode.NOT_AUTHORIZED,
            message="❌ 未查询到该订单，或您无权访问此订单。",
        )

    items_str = ", ".join([f"{i['name']}(x{i['qty']})" for i in order.items])
    message = (
        f"订单号: {order.order_sn}\n"
        f"状态: {order.status}\n"
        f"商品: {items_str}\n"
        f"金额: {order.total_amount}元\n"
        f"物流单号: {order.tracking_number or '暂无'}"
    )
    return ToolOutcome(
        ok=True,
        code=ToolCode.SUCCESS,
        message=message,
        data={
            "order_id": int(order.id),
            "order_sn": order.order_sn,
            "status": str(order.status),
            "total_amount": str(order.total_amount),
            "items": order.items,
            "tracking_number": order.tracking_number,
        },
    )


@tool
async def query_order_tool(
    question: str,
    user_id: Annotated[int | None, InjectedState("user_id")] = None,
    active_order_sn: Annotated[str | None, InjectedState("active_order_sn")] = None,
    active_domain: Annotated[str | None, InjectedState("active_domain")] = None,
) -> str:
    """查询订单信息。

    使用场景：
    - 用户询问"我的订单 X 怎么样了？"
    - 订单号已写入会话工作记忆时优先以工作记忆为准。

    返回：
    - 订单基本信息（订单号 / 状态 / 商品 / 金额 / 物流单号）
    """
    _ensure_capabilities_registered()
    outcome = await _executor.invoke(
        "query_order_tool",
        state={
            "user_id": user_id,
            "active_order_sn": active_order_sn,
            "active_domain": active_domain,
        },
        arguments={"question": question},
    )
    return outcome.message


# ==========================================================
# 工具列表导出（同时供 Registry 和 LangChain ToolNode 使用）
# ==========================================================
# 这些 LangChain 工具薄壳均通过 GuardedToolExecutor 路由到 Registry，
# 因此 ToolNode 调用与 Workflow 直接调用共享同一套 Guard 规则。
refund_tools = [
    check_refund_eligibility,
    submit_refund_application,
    query_refund_status,
]

order_tools = [query_order_tool]

__all__ = [
    "check_refund_eligibility",
    "core_check_refund_eligibility",
    "core_query_order",
    "core_query_refund_status",
    "core_submit_refund_application",
    "order_tools",
    "query_order_tool",
    "query_refund_status",
    "refund_tools",
    "submit_refund_application",
]