import asyncio
from uuid import uuid4

import pytest
from sqlmodel import select

from app.core.database import async_session_maker
from app.models.order import Order, OrderStatus
from app.models.refund import RefundApplication
from app.models.user import User
from app.services.refund_service import RefundApplicationService, RefundReason

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_concurrent_refund_submission_has_one_winner():
    suffix = uuid4().hex[:12]
    async with async_session_maker() as session:
        user = User(
            username=f"concurrency_{suffix}",
            password_hash=User.hash_password("password123"),
            email=f"concurrency_{suffix}@example.com",
            full_name="Concurrency User",
        )
        session.add(user)
        await session.flush()
        order = Order(
            order_sn=f"CONCURRENCY-{suffix}",
            user_id=user.id,
            status=OrderStatus.DELIVERED,
            total_amount=100,
            items=[{"name": "测试商品", "qty": 1, "price": 100}],
            shipping_address="test",
        )
        session.add(order)
        await session.commit()
        user_id, order_id = user.id, order.id

    async def submit():
        async with async_session_maker() as session:
            return await RefundApplicationService.create_refund_application(
                order_id=order_id,
                user_id=user_id,
                reason_detail="并发测试",
                reason_category=RefundReason.OTHER,
                session=session,
            )

    first, second = await asyncio.gather(submit(), submit())
    outcomes = [first, second]
    assert sum(result[0] for result in outcomes) == 1
    assert sum(result[2] is not None for result in outcomes) == 2

    async with async_session_maker() as session:
        refunds = (await session.exec(select(RefundApplication).where(RefundApplication.order_id == order_id))).all()
        assert len(refunds) == 1
