# app/api/v1/schemas.py
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    # 用户的问题
    question: str = Field(..., example="内衣拆封了可以退吗？")

    client_session_id: str | None = Field(
        default=None, min_length=1, max_length=128, examples=["browser_123"]
    )
    conversation_id: UUID | None = None
    # Deprecated compatibility alias. New clients must use client_session_id.
    thread_id: str | None = Field(default="default_thread", min_length=1, max_length=128)
    # P1: 用户对退款申请的显式确认标志。前端在 WAITING_CONFIRMATION 阶段展示
    # 订单/金额/原因，用户点击「确认提交」按钮后回传 user_confirmed=True；
    # 子图节点据此推进到 SUBMITTED 阶段。默认 False，不影响 ORDER/POLICY 路径。
    user_confirmed: bool = Field(default=False)

    @model_validator(mode="after")
    def require_client_session(self):
        if not self.client_session_id and not self.thread_id:
            raise ValueError("client_session_id is required")
        return self

    @property
    def resolved_client_session_id(self) -> str:
        return self.client_session_id or self.thread_id or ""

class ChatResponse(BaseModel):
    # 非流式模式下的返回结构
    answer: str
