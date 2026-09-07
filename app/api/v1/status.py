# app/api/v1/status.py
"""
状态查询 API
用于前端轮询获取 Agent 处理状态
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.core.security import get_current_user_id
from app.core.database import async_session_maker
from app.models.audit import AuditLog, AuditAction
from app.models.message import MessageCard, MessageType
from sqlmodel import select, desc

router = APIRouter()


class StatusResponse(BaseModel):
    """状态响应"""
    thread_id: str
    status:  str  # "IDLE", "WAITING_ADMIN", "APPROVED", "REJECTED", "COMPLETED", "ERROR"
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: str


@router.get("/status/{thread_id}", response_model=StatusResponse)
async def get_thread_status(
    thread_id: str,
    current_user_id: int = Depends(get_current_user_id)
):
    """
    获取会话状态
    
    用于 C 端轮询查询当前会话的处理状态
    """
    async with async_session_maker() as session:
        # 1. 查询该会话最新的审计日志
        audit_result = await session.execute(
            select(AuditLog)
            .where(AuditLog.thread_id == thread_id)
            .where(AuditLog.user_id == current_user_id)
            .order_by(desc(AuditLog.created_at))
            .limit(1)
        )
        latest_audit = audit_result.scalar_one_or_none()
        
        # 2. 查询最新消息
        message_result = await session.execute(
            select(MessageCard)
            .where(MessageCard.thread_id == thread_id)
            .order_by(desc(MessageCard.created_at))
            .limit(1)
        )
        latest_message = message_result.scalar_one_or_none()
        
        # 3. 判断状态
        if latest_audit: 
            if latest_audit.action == AuditAction.PENDING:
                return StatusResponse(
                    thread_id=thread_id,
                    status="WAITING_ADMIN",
                    message="人工审核中，请稍候.. .",
                    data={
                        "audit_log_id": latest_audit.id,
                        "risk_level": latest_audit.risk_level,
                        "trigger_reason": latest_audit.trigger_reason,
                    },
                    timestamp=latest_audit.created_at.isoformat()
                )
            elif latest_audit.action == AuditAction.APPROVE:
                return StatusResponse(
                    thread_id=thread_id,
                    status="APPROVED",
                    message="审核通过，正在处理退款.. .",
                    data={
                        "admin_comment": latest_audit.admin_comment,
                        "reviewed_at": latest_audit.reviewed_at.isoformat() if latest_audit.reviewed_at else None,
                    },
                    timestamp=latest_audit.updated_at.isoformat()
                )
            elif latest_audit.action == AuditAction.REJECT:
                return StatusResponse(
                    thread_id=thread_id,
                    status="REJECTED",
                    message=f"审核未通过:  {latest_audit.admin_comment or '请联系客服'}",
                    data={
                        "admin_comment":  latest_audit.admin_comment,
                        "reviewed_at": latest_audit.reviewed_at.isoformat() if latest_audit.reviewed_at else None,
                    },
                    timestamp=latest_audit.updated_at.isoformat()
                )
        
        # 4. 无审核记录：当前没有进行中的退款/审核事件，返回中性 IDLE 态。
        # 此前这里无条件兜底 PROCESSING，导致前端在查订单、问政策等任意轮次
        # 都把"退款处理中"横幅拼进机器人回复。PROCESSING 只能来自退款申请的
        # 业务状态（RefundStatus），不是本接口的兜底值。
        return StatusResponse(
            thread_id=thread_id,
            status="IDLE",
            message="当前无进行中的售后事项",
            data={},
            timestamp=latest_message.created_at.isoformat() if latest_message else ""
        )