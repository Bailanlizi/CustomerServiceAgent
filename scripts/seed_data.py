"""Create idempotent demo users and orders for the local development database."""

import asyncio
import os
import sys

sys.path.append(os.getcwd())

from sqlmodel import select

from app.core.database import async_session_maker
from app.models.order import Order, OrderStatus
from app.models.user import User


DEMO_USERS = (
    {"username": "alice", "password": "alice123", "email": "alice@example.com", "full_name": "Alice Wang", "is_admin": False},
    {"username": "bob", "password": "bob123", "email": "bob@example.com", "full_name": "Bob Li", "is_admin": False},
    {"username": "admin", "password": "admin123", "email": "admin@example.com", "full_name": "System Admin", "is_admin": True},
)

DEMO_ORDERS = (
    {"order_sn": "SN20240001", "username": "alice", "status": OrderStatus.SHIPPED, "total_amount": 128.50, "items": [{"name": "运动内衣", "qty": 1, "price": 128.50}], "tracking_number": "SF123456789", "shipping_address": "上海市浦东新区张江高科技园区"},
    {"order_sn": "SN20240002", "username": "alice", "status": OrderStatus.PENDING, "total_amount": 50.00, "items": [{"name": "全棉袜子", "qty": 5, "price": 10.00}], "tracking_number": None, "shipping_address": "上海市浦东新区张江高科技园区"},
    {"order_sn": "SN20240003", "username": "alice", "status": OrderStatus.SHIPPED, "total_amount": 199.00, "items": [{"name": "运动T恤", "qty": 1, "price": 99.00}, {"name": "运动短裤", "qty": 1, "price": 100.00}], "tracking_number": "SF987654321", "shipping_address": "上海市浦东新区张江高科技园区"},
    {"order_sn": "SN20240004", "username": "bob", "status": OrderStatus.DELIVERED, "total_amount": 599.00, "items": [{"name": "耐克篮球鞋", "qty": 1, "price": 599.00}], "tracking_number": "SF555666777", "shipping_address": "北京市海淀区中关村"},
    {"order_sn": "SN20240005", "username": "bob", "status": OrderStatus.PAID, "total_amount": 89.00, "items": [{"name": "速干运动袜", "qty": 2, "price": 44.50}], "tracking_number": None, "shipping_address": "北京市海淀区中关村"},
)


async def seed_data() -> None:
    async with async_session_maker() as session:
        users: dict[str, User] = {}
        for data in DEMO_USERS:
            user = (await session.exec(select(User).where(User.username == data["username"]))).first()
            if user is None:
                user = User(
                    username=data["username"],
                    password_hash=User.hash_password(data["password"]),
                    email=data["email"],
                    full_name=data["full_name"],
                    is_admin=data["is_admin"],
                )
                session.add(user)
                await session.flush()
                print(f"Created user: {user.username}")
            users[data["username"]] = user

        for data in DEMO_ORDERS:
            order = (await session.exec(select(Order).where(Order.order_sn == data["order_sn"]))).first()
            if order is None:
                session.add(
                    Order(
                        order_sn=data["order_sn"],
                        user_id=users[data["username"]].id,
                        status=data["status"],
                        total_amount=data["total_amount"],
                        items=data["items"],
                        tracking_number=data["tracking_number"],
                        shipping_address=data["shipping_address"],
                    )
                )
                print(f"Created order: {data['order_sn']}")

        await session.commit()
        print("Demo data is ready.")


if __name__ == "__main__":
    asyncio.run(seed_data())
