# app/api/v1/admin.py
"""
管理员 API
"""
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, model_validator
from sqlmodel import desc, select

from app.core.database import async_session_maker
from app.core.security import get_admin_user_id
from app.models.audit import AuditAction, AuditLog
from app.models.conversation import ConversationSession
from app.models.message import MessageCard, MessageStatus, MessageType
from app.models.refund import RefundApplication, RefundStatus
from app.tasks.refund_tasks import process_refund_payment, send_refund_sms
from app.websocket.manager import manager

router = APIRouter()


class AuditTask(BaseModel):
    """审核任务"""
    audit_log_id: int
    thread_id: str
    user_id: int
    refund_application_id: int | None
    order_id: int | None
    trigger_reason: str
    risk_level: str
    context_snapshot: dict[str, Any]
    created_at: str
    conversation_summary: str | None = None
    active_order_sn: str | None = None
    refund_reason: str | None = None
    refund_reason_category: str | None = None
    # P2: 结构化元数据（PLAN 第 3 / 5 节）
    eligibility_result: dict[str, Any] | None = None
    last_tool_outcome: dict[str, Any] | None = None
    refund_amount: str | None = None
    refund_risk_level: str | None = None
    refund_status: str | None = None
    refund_application_id_extra: int | None = None
    # 会话活跃工作记忆摘要（audit 字段以外的结构化摘要）
    workflow_stage: str | None = None
    user_confirmed: bool | None = None


class AdminDecisionRequest(BaseModel):
    """管理员决策请求"""
    action: Literal["APPROVE", "REJECT"]
    admin_comment: str | None = None

    @model_validator(mode="after")
    def reject_requires_comment(self):
        if self.action == "REJECT" and not (self.admin_comment or "").strip():
            raise ValueError("拒绝退款时必须填写审核备注")
        return self


class AdminDecisionResponse(BaseModel):
    """管理员决策响应"""
    success: bool
    message: str
    audit_log_id: int
    action: str


@router.get("/admin/tasks", response_model=list[AuditTask])
async def get_pending_tasks(
    risk_level: str | None = None,
    current_admin_id: int = Depends(get_admin_user_id)
):
    """
    获取待审核任务列表
    
    Query Params:
        risk_level: 可选，筛选风险等级 (HIGH, MEDIUM, LOW)
    """
    async with async_session_maker() as session:
        # 构建查询
        stmt = select(AuditLog).where(
            AuditLog.action == AuditAction.PENDING
        ).order_by(desc(AuditLog.created_at))
        
        if risk_level: 
            stmt = stmt.where(AuditLog.risk_level == risk_level)
        
        result = await session.execute(stmt)
        audit_logs = result.scalars().all()

        # P1-2 收尾：批量预取 ConversationSession，消除 N+1 查询。
        # 之前每条 audit_log 单独查询 conversation，任务量大时放大 DB 压力；
        # 改为先收集 thread_ids，一次 select，再用 dict 查找。
        thread_ids = {log.thread_id for log in audit_logs if log.thread_id}
        conversations_by_thread: dict[str, Any] = {}
        if thread_ids:
            conv_result = await session.execute(
                select(ConversationSession).where(
                    ConversationSession.checkpoint_thread_id.in_(thread_ids)
                )
            )
            conversations_by_thread = {
                c.checkpoint_thread_id: c
                for c in conv_result.scalars().all()
                if c.checkpoint_thread_id
            }

        # 转换为响应格式
        tasks = []
        for log in audit_logs:
            conversation = conversations_by_thread.get(log.thread_id)
            memory = conversation.working_memory_json if conversation else {}
            slots = memory.get("collected_slots") if isinstance(memory, dict) else {}
            slots = slots if isinstance(slots, dict) else {}
            last_result = memory.get("last_tool_result") if isinstance(memory, dict) else None
            tool_outcome = last_result.get("tool_outcome") if isinstance(last_result, dict) else None
            eligibility = (
                {k: last_result.get(k) for k in ("eligibility_checked", "eligibility_passed", "eligibility_message")}
                if isinstance(last_result, dict) else None
            )
            # P2: 独立暴露 refund_amount / risk_level，源自最近一次 ToolOutcome.data。
            # 管理员审批无需重新询问用户即可获得核心决策数据。
            outcome_data = (tool_outcome or {}).get("data") if isinstance(tool_outcome, dict) else None
            refund_amount = None
            refund_risk_level = None
            refund_status = None
            if isinstance(outcome_data, dict):
                if "refund_amount" in outcome_data:
                    refund_amount = str(outcome_data.get("refund_amount"))
                if "risk_level" in outcome_data:
                    refund_risk_level = str(outcome_data.get("risk_level"))
                if "status" in outcome_data:
                    refund_status = str(outcome_data.get("status"))
            # 回退到 context_snapshot（refund 工具在成功路径写入）
            if refund_amount is None and isinstance(log.context_snapshot, dict):
                refund_block = log.context_snapshot.get("refund") or {}
                if isinstance(refund_block, dict) and refund_block.get("refund_amount"):
                    refund_amount = str(refund_block["refund_amount"])
            tasks.append(AuditTask(
                audit_log_id=log.id,
                thread_id=log.thread_id,
                user_id=log.user_id,
                refund_application_id=log.refund_application_id,
                order_id=log.order_id,
                trigger_reason=log.trigger_reason,
                risk_level=log.risk_level,
                context_snapshot=log.context_snapshot,
                created_at=log.created_at.isoformat(),
                conversation_summary=conversation.conversation_summary if conversation else None,
                active_order_sn=memory.get("active_order_sn") if isinstance(memory, dict) else None,
                refund_reason=slots.get("refund_reason"),
                refund_reason_category=memory.get("refund_reason_category") if isinstance(memory, dict) else None,
                eligibility_result=eligibility,
                last_tool_outcome=tool_outcome,
                refund_amount=refund_amount,
                refund_risk_level=refund_risk_level,
                refund_status=refund_status,
                refund_application_id_extra=log.refund_application_id,
                workflow_stage=memory.get("workflow_stage") if isinstance(memory, dict) else None,
                user_confirmed=bool(slots.get("user_confirmed")) if isinstance(slots, dict) else None,
            ))

        return tasks


@router.post("/admin/resume/{audit_log_id}", response_model=AdminDecisionResponse)
async def admin_decision(
    audit_log_id: int,
    request: AdminDecisionRequest,
    current_admin_id: int = Depends(get_admin_user_id)
):
    """
    管理员决策接口
    
    Path Params:
        audit_log_id: 审计日志ID
    
    Body:
        action:  APPROVE | REJECT
        admin_comment:  管理员备注
    """
    async with async_session_maker() as session:
        # 始终先锁审核记录，再锁退款记录，确保并发审批顺序一致。
        result = await session.execute(
            select(AuditLog).where(AuditLog.id == audit_log_id).with_for_update()
        )
        audit_log = result.scalar_one_or_none()
        
        if not audit_log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audit log not found"
            )
        
        if audit_log.action != AuditAction.PENDING: 
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="该申请已被处理"
            )

        if not audit_log.refund_application_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="审核任务未关联退款申请",
            )

        refund_result = await session.execute(
            select(RefundApplication)
            .where(RefundApplication.id == audit_log.refund_application_id)
            .with_for_update()
        )
        refund = refund_result.scalar_one_or_none()
        if not refund:
            raise HTTPException(status_code=404, detail="Refund application not found")
        if refund.status != RefundStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="该申请已被处理",
            )

        action_enum = AuditAction.APPROVE if request.action == "APPROVE" else AuditAction.REJECT
        reviewed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        audit_log.action = action_enum
        audit_log.admin_id = current_admin_id
        audit_log.admin_comment = request.admin_comment
        audit_log.reviewed_at = reviewed_at
        session.add(audit_log)

        refund.status = (
            RefundStatus.APPROVED if action_enum == AuditAction.APPROVE else RefundStatus.REJECTED
        )
        refund.admin_note = request.admin_comment
        refund.reviewed_by = current_admin_id
        refund.reviewed_at = reviewed_at
        session.add(refund)
        
        # 5. 创建状态变更消息卡片
        status_message = (
            "审核通过，退款已进入处理队列"
            if action_enum == AuditAction.APPROVE
            else f"审核未通过: {request.admin_comment}"
        )
        
        message_card = MessageCard(
            thread_id=audit_log.thread_id,
            message_type=MessageType.AUDIT_CARD,
            status=MessageStatus.SENT,
            content={
                "card_type": "audit_result",
                "action": request.action,
                "message": status_message,
                "admin_comment": request.admin_comment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            sender_type="admin",
            sender_id=current_admin_id,
            receiver_id=audit_log.user_id,
        )
        session.add(message_card)
        
        await session.commit()

        dispatch_errors: list[str] = []
        if action_enum == AuditAction.APPROVE:
            try:
                process_refund_payment.delay(refund_id=refund.id)
            except Exception as exc:
                dispatch_errors.append(f"payment: {exc}")
            try:
                send_refund_sms.delay(
                    refund_id=refund.id,
                    phone="138****1234",  # TODO: 从用户表获取
                    message=f"您的退款申请已通过，退款金额¥{refund.refund_amount}已进入处理队列。",
                )
            except Exception as exc:
                dispatch_errors.append(f"sms: {exc}")

        try:
            await manager.notify_status_change(
                thread_id=audit_log.thread_id,
                status=request.action,
                data={"message": status_message, "admin_comment": request.admin_comment},
            )
        except Exception as exc:
            dispatch_errors.append(f"websocket: {exc}")

        if dispatch_errors:
            async with async_session_maker() as error_session:
                error_audit = await error_session.get(AuditLog, audit_log_id)
                if error_audit:
                    metadata = dict(error_audit.decision_metadata or {})
                    metadata["dispatch_errors"] = dispatch_errors
                    metadata["dispatch_failed_at"] = datetime.now(timezone.utc).isoformat()
                    error_audit.decision_metadata = metadata
                    error_session.add(error_audit)
                    await error_session.commit()
        
        return AdminDecisionResponse(
            success=True,
            message=f"审核决策已提交:  {request.action}",
            audit_log_id=audit_log_id,
            action=request.action,
        )
