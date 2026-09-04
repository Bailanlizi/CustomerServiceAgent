"""add conversation sessions

Revision ID: c1a7f0e2b9d4
Revises: b7d31a8c4e20
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "c1a7f0e2b9d4"
down_revision: Union[str, None] = "b7d31a8c4e20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conversation_sessions",
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("client_session_id", sa.String(length=128), nullable=False),
        sa.Column("checkpoint_thread_id", sa.String(length=128), nullable=False),
        sa.Column("active_domain", sa.String(length=16), nullable=True),
        sa.Column("working_memory_json", sa.JSON(), nullable=False),
        sa.Column("conversation_summary", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("last_active_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("conversation_id"),
        sa.UniqueConstraint("checkpoint_thread_id"),
        sa.UniqueConstraint("user_id", "client_session_id", name="uq_conversation_user_client"),
    )
    op.create_index("ix_conversation_sessions_user_id", "conversation_sessions", ["user_id"])
    op.create_index("ix_conversation_sessions_active_domain", "conversation_sessions", ["active_domain"])
    op.create_index("ix_conversation_sessions_checkpoint_thread_id", "conversation_sessions", ["checkpoint_thread_id"], unique=True)
    op.create_index("ix_conversation_sessions_last_active_at", "conversation_sessions", ["last_active_at"])


def downgrade() -> None:
    op.drop_index("ix_conversation_sessions_last_active_at", table_name="conversation_sessions")
    op.drop_index("ix_conversation_sessions_checkpoint_thread_id", table_name="conversation_sessions")
    op.drop_index("ix_conversation_sessions_active_domain", table_name="conversation_sessions")
    op.drop_index("ix_conversation_sessions_user_id", table_name="conversation_sessions")
    op.drop_table("conversation_sessions")
