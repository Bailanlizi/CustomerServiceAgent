# app/tasks/refund_tasks.py
"""
退款相关异步任务
"""
import asyncio
import time
from typing import Dict, Any
from celery import Task
from app.celery_app import celery_app
from app.core.database import async_session_maker
from app.models.refund import RefundApplication, RefundStatus
from app.models.audit import AuditLog
from app.models.message import MessageCard, MessageType, MessageStatus
from sqlmodel import select
from sqlalchemy import update
from datetime import datetime, timedelta, timezone
from app.core.config import settings


class DatabaseTask(Task):
    """支持异步数据库操作的 Celery Task 基类"""
    _session = None
    
    def run_async(self, coro):
        """在 Celery worker 中运行异步函数"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="refund.send_sms",
    max_retries=3,
    default_retry_delay=60
)
def send_refund_sms(self, refund_id: int, phone:  str, message: str) -> Dict[str, Any]:
    """
    发送退款通知短信
    
    Args:
        refund_id:  退款申请ID
        phone: 手机号
        message: 短信内容
    """
    try:
        # TODO: 接入真实短信网关 (阿里云、腾讯云等)
        print(f"📱 [SMS] 发送短信到 {phone}: {message}")
        
        # 模拟短信发送
        import time
        time.sleep(2)
        
        # 记录发送成功
        return {
            "status": "success",
            "refund_id": refund_id,
            "phone": phone,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as exc:
        # 重试机制
        print(f"  [SMS] 发送失败: {exc}")
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="refund.process_payment",
    max_retries=3,
    default_retry_delay=120
)
def process_refund_payment(self, refund_id: int) -> Dict[str, Any]:
    """
    调用支付网关执行退款
    
    Args:
        refund_id:  退款申请ID
        金额始终从数据库读取，调用方不能覆盖。
    """
    async def _claim():
        async with async_session_maker() as session:
            result = await session.execute(
                update(RefundApplication)
                .where(
                    RefundApplication.id == refund_id,
                    RefundApplication.status == RefundStatus.APPROVED,
                )
                .values(
                    status=RefundStatus.PROCESSING,
                    updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
                .returning(RefundApplication.refund_amount)
            )
            amount = result.scalar_one_or_none()
            await session.commit()
            if amount is not None:
                return "claimed", amount

            refund = await session.get(RefundApplication, refund_id)
            if not refund:
                raise ValueError(f"Refund application {refund_id} not found")
            if refund.status in (RefundStatus.PROCESSING, RefundStatus.COMPLETED):
                return "noop", refund.refund_amount
            return "invalid", refund.refund_amount

    async def _complete():
        async with async_session_maker() as session:
            result = await session.execute(
                update(RefundApplication)
                .where(
                    RefundApplication.id == refund_id,
                    RefundApplication.status == RefundStatus.PROCESSING,
                )
                .values(
                    status=RefundStatus.COMPLETED,
                    updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            if result.rowcount != 1:
                await session.rollback()
                raise RuntimeError(f"Refund {refund_id} left PROCESSING unexpectedly")
            await session.commit()

    async def _restore_after_failure(exc: Exception):
        async with async_session_maker() as session:
            await session.execute(
                update(RefundApplication)
                .where(
                    RefundApplication.id == refund_id,
                    RefundApplication.status == RefundStatus.PROCESSING,
                )
                .values(
                    status=RefundStatus.APPROVED,
                    updated_at=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            audit_result = await session.execute(
                select(AuditLog).where(AuditLog.refund_application_id == refund_id)
            )
            audit_log = audit_result.scalars().first()
            if audit_log:
                metadata = dict(audit_log.decision_metadata or {})
                metadata.update({
                    "payment_error": str(exc),
                    "payment_failed_at": datetime.now(timezone.utc).isoformat(),
                    "payment_retry": self.request.retries,
                })
                audit_log.decision_metadata = metadata
                session.add(audit_log)
            await session.commit()

    try:
        claim_status, amount = self.run_async(_claim())
        if claim_status == "noop":
            return {"status": "noop", "refund_id": refund_id}
        if claim_status == "invalid":
            return {"status": "rejected", "refund_id": refund_id}

        print(f"💰 [Payment] 退款 ¥{amount} 到原支付方式")
        time.sleep(3)
        self.run_async(_complete())
        return {
            "status": "success",
            "refund_id": refund_id,
            "amount": float(amount),
            "transaction_id": f"TXN{refund_id}{int(time.time())}",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        print(f"  [Payment] 退款失败: {exc}")
        self.run_async(_restore_after_failure(exc))
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="refund.recover_stalled",
)
def recover_stalled_refunds(self) -> Dict[str, Any]:
    """Recover refunds abandoned in PROCESSING by a crashed worker."""
    async def _recover():
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            minutes=settings.REFUND_PROCESSING_TIMEOUT_MINUTES
        )
        async with async_session_maker() as session:
            result = await session.execute(
                select(RefundApplication)
                .where(
                    RefundApplication.status == RefundStatus.PROCESSING,
                    RefundApplication.updated_at < cutoff,
                )
                .with_for_update(skip_locked=True)
            )
            refunds = result.scalars().all()
            recovered_ids = []
            for refund in refunds:
                refund.status = RefundStatus.APPROVED
                refund.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                session.add(refund)
                audit_result = await session.execute(
                    select(AuditLog).where(AuditLog.refund_application_id == refund.id)
                )
                audit_log = audit_result.scalars().first()
                if audit_log:
                    metadata = dict(audit_log.decision_metadata or {})
                    metadata.update({
                        "payment_error": "退款处理超时，已恢复为待重新处理状态",
                        "payment_failed_at": datetime.now(timezone.utc).isoformat(),
                        "payment_recovered_by": "refund.recover_stalled",
                    })
                    audit_log.decision_metadata = metadata
                    session.add(audit_log)
                recovered_ids.append(refund.id)
            await session.commit()
            return recovered_ids

    recovered_ids = self.run_async(_recover())
    requeue_errors = {}
    for refund_id in recovered_ids:
        try:
            process_refund_payment.delay(refund_id=refund_id)
        except Exception as exc:
            # Keep APPROVED so the next scheduled sweep can safely retry dispatch.
            requeue_errors[str(refund_id)] = str(exc)
    return {
        "status": "success",
        "recovered_refund_ids": recovered_ids,
        "requeue_errors": requeue_errors,
    }


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    name="refund.notify_admin",
    max_retries=2
)
def notify_admin_audit(self, audit_log_id: int) -> Dict[str, Any]: 
    """
    通知管理员有新的审核任务
    
    Args: 
        audit_log_id: 审计日志ID
    """
    async def _notify():
        async with async_session_maker() as session:
            # 查询审计日志
            result = await session.execute(
                select(AuditLog).where(AuditLog.id == audit_log_id)
            )
            audit_log = result.scalar_one_or_none()
            
            if not audit_log: 
                raise ValueError(f"Audit log {audit_log_id} not found")
            
            # TODO: 接入真实通知系统 (邮件、企业微信、钉钉等)
            print("  [Notify] 通知管理员审核任务:")
            print(f"  - 风险等级: {audit_log.risk_level}")
            print(f"  - 触发原因: {audit_log.trigger_reason}")
            print(f"  - 用户ID: {audit_log.user_id}")
            
            # 创建系统消息通知 B 端
            message = MessageCard(
                thread_id=audit_log.thread_id,
                message_type=MessageType.SYSTEM,
                status=MessageStatus.SENT,
                content={
                    "type": "admin_notification",
                    "audit_log_id": audit_log_id,
                    "risk_level": audit_log.risk_level,
                    "message": f"新的{audit_log.risk_level}风险审核任务",
                },
                sender_type="system",
                receiver_id=None,  # 广播给所有管理员
            )
            session.add(message)
            await session.commit()
            
            return {
                "status": "success",
                "audit_log_id": audit_log_id,
                "notified_at": datetime.now(timezone.utc).isoformat(),
            }
    
    try:
        return self.run_async(_notify())
    except Exception as exc:
        print(f"  [Notify] 通知失败:  {exc}")
        raise self.retry(exc=exc)
