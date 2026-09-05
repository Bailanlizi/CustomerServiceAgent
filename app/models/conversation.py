from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlmodel import Field, SQLModel


def utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ConversationSession(SQLModel, table=True):
    """Persistent business-session state; checkpoints remain in Redis."""

    __tablename__ = "conversation_sessions"
    __table_args__ = (
        UniqueConstraint("user_id", "client_session_id", name="uq_conversation_user_client"),
    )

    conversation_id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: int = Field(foreign_key="users.id", index=True, ondelete="RESTRICT")
    client_session_id: str = Field(max_length=128)
    checkpoint_thread_id: str = Field(unique=True, index=True, max_length=128)
    active_domain: str | None = Field(
        default=None, sa_column=Column(String(16), nullable=True, index=True)
    )
    working_memory_json: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    conversation_summary: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    version: int = Field(default=1, nullable=False)
    last_active_at: datetime = Field(
        default_factory=utcnow_naive,
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"), index=True),
    )
    created_at: datetime = Field(
        default_factory=utcnow_naive,
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    )
    updated_at: datetime = Field(
        default_factory=utcnow_naive,
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
            onupdate=text("CURRENT_TIMESTAMP"),
        ),
    )


class ConversationMessage(SQLModel, table=True):
    """Persistent user-visible transcript for restoring the customer chat UI."""

    __tablename__ = "conversation_messages"
    __table_args__ = (
        Index("ix_conversation_messages_conversation_created", "conversation_id", "created_at", "id"),
        Index("ix_conversation_messages_user_created", "user_id", "created_at"),
    )

    id: int | None = Field(default=None, primary_key=True)
    conversation_id: UUID = Field(
        sa_column=Column(ForeignKey("conversation_sessions.conversation_id", ondelete="CASCADE"), nullable=False, index=True)
    )
    user_id: int = Field(foreign_key="users.id", index=True)
    role: str = Field(max_length=16)
    content: str = Field(sa_column=Column(Text, nullable=False))
    message_type: str = Field(default="text", max_length=32)
    created_at: datetime = Field(
        default_factory=utcnow_naive,
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    )
