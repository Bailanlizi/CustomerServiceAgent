"""secure refund workflow

Revision ID: b7d31a8c4e20
Revises: 6a81cd1a4aba
Create Date: 2026-09-04
"""

from typing import Sequence, Union

from alembic import op


revision: str = "b7d31a8c4e20"
down_revision: Union[str, None] = "6a81cd1a4aba"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A physical UNIQUE constraint cannot retain multiple rows for one order, even
    # when later rows are CANCELLED. Preserve their IDs/statuses on the retained
    # case, then remove the mock-only duplicate rows before adding the constraint.
    op.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                first_value(id) OVER (
                    PARTITION BY order_id ORDER BY created_at ASC, id ASC
                ) AS kept_id,
                row_number() OVER (
                    PARTITION BY order_id ORDER BY created_at ASC, id ASC
                ) AS duplicate_rank
            FROM refund_applications
        )
        UPDATE audit_logs AS audit
        SET refund_application_id = ranked.kept_id,
            updated_at = CURRENT_TIMESTAMP
        FROM ranked
        WHERE audit.refund_application_id = ranked.id
          AND ranked.duplicate_rank > 1
        """
    )
    op.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                first_value(id) OVER (
                    PARTITION BY order_id ORDER BY created_at ASC, id ASC
                ) AS kept_id,
                row_number() OVER (
                    PARTITION BY order_id ORDER BY created_at ASC, id ASC
                ) AS duplicate_rank
            FROM refund_applications
        ), duplicate_summary AS (
            SELECT
                kept_id,
                string_agg(
                    '已清理历史重复退款案件 ID=' || refund.id ||
                    '（原状态=' || refund.status || '，按 CANCELLED 归档）',
                    E'\n' ORDER BY refund.created_at, refund.id
                ) AS cleanup_note
            FROM ranked
            JOIN refund_applications AS refund ON refund.id = ranked.id
            WHERE ranked.duplicate_rank > 1
            GROUP BY kept_id
        )
        UPDATE refund_applications AS kept
        SET admin_note = concat_ws(
                E'\n', nullif(kept.admin_note, ''), duplicate_summary.cleanup_note
            ),
            updated_at = CURRENT_TIMESTAMP
        FROM duplicate_summary
        WHERE kept.id = duplicate_summary.kept_id
        """
    )
    op.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                row_number() OVER (
                    PARTITION BY order_id ORDER BY created_at ASC, id ASC
                ) AS duplicate_rank
            FROM refund_applications
        )
        DELETE FROM refund_applications AS refund
        USING ranked
        WHERE refund.id = ranked.id
          AND ranked.duplicate_rank > 1
        """
    )
    op.drop_index("ix_refund_applications_order_id", table_name="refund_applications")
    op.create_unique_constraint(
        "uq_refund_applications_order_id",
        "refund_applications",
        ["order_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_refund_applications_order_id",
        "refund_applications",
        type_="unique",
    )
    op.create_index(
        "ix_refund_applications_order_id",
        "refund_applications",
        ["order_id"],
        unique=False,
    )
