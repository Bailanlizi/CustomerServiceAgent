# app/graph/tools.py
"""
LangGraph Tools: Agent 可调用的工具函数
"""
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field
from sqlmodel import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.audit import AuditAction, AuditLog, RiskLevel
from app.models.order import Order
from app.services.refund_service import (
    RefundApplicationService,
    RefundEligibilityChecker,
    RefundReason,
)
from app.tasks.refund_tasks import notify_admin_audit
from app.websocket.manager import manager

# ==========================================
# 工具 1: 检查退货资格
# ==========================================

@tool
async def check_refund_eligibility(
    # P1 修复: 订单号同样改为 InjectedState，来源是 ConversationStateManager
    # 已确认的 active_order_sn，避免 LLM 在工具调用时重新声明、与工作记忆不一致。
    # 标 Optional 是因为工作记忆缺值时 Pydantic 会注入 None，前置校验负责给出
    # 用户友好的拒绝信息，而不是被工具签名校验打断。
    order_sn: Annotated[str | None, InjectedState("active_order_sn")] = None,
    user_id: Annotated[int | None, InjectedState("user_id")] = None,
) -> str:
    """
    检查订单是否符合退货条件。

    使用场景：
    - 用户询问"我的订单能退货吗？"
    - 在正式申请退货前进行资格预检

    前置约束：
    - 订单号必须已写入会话工作记忆（active_order_sn），否则提示用户补全。

    返回：
    - 如果可以退货，返回"符合退货条件"及详细说明
    - 如果不能退货，返回拒绝原因（如：超期、已退、商品类别等）
    """
    if not order_sn:
        return "❌ 缺少订单号，无法进行资格预检。请先告知要查询的订单号。"
    if not user_id:
        return "❌ 缺少用户身份，无法进行资格预检。"

    async with async_session_maker() as session:
        # 1. 查询订单（带用户权限校验）
        stmt = select(Order).where(
            Order.order_sn == order_sn,
            Order.user_id == user_id  # 🔒 安全：只能查自己的订单
        )
        result = await session.exec(stmt)
        order = result.first()

        if not order:
            return f"❌ 未找到订单 {order_sn}，或您无权访问此订单。"
        
        # 2. 调用规则引擎检查资格
        is_eligible, message = await RefundEligibilityChecker.check_eligibility(
            order, session
        )
        
        # 3. 格式化返回结果
        if is_eligible:
            return (
                f"✅ 订单 {order_sn} 符合退货条件。\n"
                f"订单信息：\n"
                f"  - 商品：{', '.join([item['name'] for item in order.items])}\n"
                f"  - 金额：¥{order.total_amount}\n"
                f"  - 状态：{order.status}\n"
                f"检查结果：{message}"
            )
        else:
            return (
                f"❌ 订单 {order_sn} 不符合退货条件。\n"
                f"拒绝原因：{message}"
            )


# ==========================================
# 工具 2: 提交退货申请
# ==========================================

@tool
async def submit_refund_application(
    # P1 修复: 订单号、退款原因与分类由 ConversationStateManager 写入 working memory，
    # 这里通过 InjectedState 直接消费；不再让 LLM 自由声明参数，杜绝"工作记忆里有
    # 订单号但工具仍要求重新输入/拼写不一致"的接口鸿沟。
    order_sn: Annotated[str | None, InjectedState("active_order_sn")] = None,
    user_id: Annotated[int | None, InjectedState("user_id")] = None,
    thread_id: Annotated[str | None, InjectedState("thread_id")] = None,
    reason_detail: Annotated[dict | None, InjectedState("collected_slots")] = None,
    reason_category: Annotated[
        str | None,
        InjectedState("refund_reason_category"),
    ] = None,
) -> str:
    """
    提交退货申请。

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
    # ========== 前置校验: 订单号 ==========
    if not order_sn:
        return "❌ 缺少订单号，无法提交退款申请。请先告知要退货的订单号。"
    # ========== 前置校验: 用户身份 ==========
    if not user_id:
        return "❌ 缺少用户身份，无法提交退款申请。"
    # ========== 前置校验: 退款原因 ==========
    slots = reason_detail if isinstance(reason_detail, dict) else {}
    reason_text = slots.get("refund_reason") if isinstance(slots, dict) else None
    if not reason_text:
        return "❌ 缺少退货原因，无法提交退款申请。请先说明退货原因（如质量问题、尺码不合适等）。"

    async with async_session_maker() as session:
        # 1. 查询订单
        stmt = select(Order).where(
            Order.order_sn == order_sn,
            Order.user_id == user_id  # 🔒 安全校验
        )
        result = await session.exec(stmt)
        order = result.first()

        if not order:
            return f"❌ 未找到订单 {order_sn}，或您无权访问此订单。"

        # 2. 转换原因分类（直接信任 working memory 中的 enum 字面量）
        category = None
        if reason_category:
            try:
                category = RefundReason(reason_category)
            except ValueError:
                category = RefundReason.OTHER

        # 3. 创建退货申请（内部会自动校验资格）
        success, message, refund_app = await RefundApplicationService.create_refund_application(
            order_id=order.id,
            user_id=user_id,
            reason_detail=reason_text,
            reason_category=category,
            session=session,
            commit=False,
        )
        
        # 4. 格式化返回结果
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
                    # reason_text 是用户填写的退货原因文本，audit 必须存原文而不是
                    # collected_slots dict（否则审计回溯会变成结构化数据，违反监管
                    # 对"原始诉求"可读性的要求）。
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
                    data={"risk_level": risk_level, "audit_log_id": audit_log.id, "refund_amount": refund_amount},
                )
            except Exception as exc:  # noqa: BLE001 - 通知失败不能回滚已写入的申请和审计。
                # 通知失败不能回滚已经创建的退款申请和审计记录。
                print(f"[Refund tool] 审核通知失败: {exc}")
            return (
                f"⏳ 退货申请已提交，需人工审核。\n"
                f"申请编号：#{refund_app.id}\n退款金额：¥{refund_amount}\n"
                f"风险等级：{risk_level}\n触发原因：{trigger_reason}\n"
                "申请已进入人工审核，请稍后查询进度。"
            )
        if refund_app:
            return f"❌ 该订单已有退款申请。\n{message}"
        return f"❌ 退货申请失败。\n原因：{message}"


# ==========================================
# 工具 3: 查询退货申请状态
# ==========================================

@tool
async def query_refund_status(
    user_id: Annotated[int, InjectedState("user_id")],
    refund_id: Annotated[
        int | None, 
        Field(description="退货申请编号，如果不提供则返回用户所有退货申请")
    ] = None
) -> str:
    """
    查询退货申请状态。
    
    使用场景：
    - 用户询问"我的退货申请怎么样了？"
    - 用户提供申请编号查询具体状态
    - 用户想查看所有退货记录
    
    返回：
    - 如果指定申请编号：返回该申请的详细信息
    - 如果未指定：返回用户所有退货申请列表
    """
    async with async_session_maker() as session:
        # 场景 1: 查询指定申请
        if refund_id:
            refund = await RefundApplicationService.get_refund_by_id(
                refund_id=refund_id,
                user_id=user_id,
                session=session
            )
            
            if not refund:
                return f"❌ 未找到申请编号 #{refund_id}，或您无权访问此申请。"
            
            # 查询关联订单信息
            stmt = select(Order).where(Order.id == refund.order_id)
            result = await session.exec(stmt)
            order = result.first()

            review_info = (
                "审核信息：\n  - 审核时间："
                + refund.reviewed_at.strftime("%Y-%m-%d %H:%M")
                if refund.reviewed_at
                else "⏳ 审核中，请耐心等待"
            )
            review_note = f"  - 审核备注：{refund.admin_note}" if refund.admin_note else ""

            return (
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
            )
        
        # 场景 2: 查询所有申请
        else:
            refund_list = await RefundApplicationService.get_user_refund_applications(
                user_id=user_id,
                session=session
            )
            
            if not refund_list:
                return "📭 您还没有退货申请记录。"
            
            result_text = f"📋 您的退货申请列表（共 {len(refund_list)} 条）\n\n"
            
            for refund in refund_list: 
                # 查询关联订单
                stmt = select(Order).where(Order.id == refund.order_id)
                order_result = await session.exec(stmt)
                order = order_result.first()
                
                status_emoji = {
                    "PENDING": "⏳",
                    "APPROVED": "✅",
                    "PROCESSING": "💳",
                    "REJECTED": "❌",
                    "COMPLETED": "🎉",
                    "CANCELLED": "🚫"
                }.get(refund.status, "❓")
                
                result_text += (
                    f"{status_emoji} 申请 #{refund.id}\n"
                    f"  订单号：{order.order_sn if order else '未知'}\n"
                    f"  状态：{refund.status}\n"
                    f"  金额：¥{refund.refund_amount}\n"
                    f"  申请时间：{refund.created_at.strftime('%Y-%m-%d')}\n\n"
                )
            
            return result_text.strip()


# ==========================================
# 工具列表导出
# ==========================================

# 将所有工具放入列表，供 LangGraph 使用
refund_tools = [
    check_refund_eligibility,
    submit_refund_application,
    query_refund_status
]
