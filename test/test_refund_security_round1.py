import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
import jwt

from app.api.v1.admin import AdminDecisionRequest
from app.core import security
from app.frontend.admin_dashboard import AdminClient
from app.models.refund import RefundStatus
from app.tasks import refund_tasks


class _FakeSession:
    def __init__(self, user):
        self.user = user

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, _model, _user_id):
        return self.user


@pytest.mark.asyncio
async def test_admin_auth_ignores_claim_and_checks_database(monkeypatch):
    ordinary_user = SimpleNamespace(is_admin=False, is_active=True)
    monkeypatch.setattr(security, "async_session_maker", lambda: _FakeSession(ordinary_user))
    forged_claim_token = jwt.encode(
        {"sub": "7", "is_admin": True}, security.settings.SECRET_KEY, algorithm=security.ALGORITHM
    )

    with pytest.raises(HTTPException) as exc_info:
        await security.get_admin_user_id(forged_claim_token)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_admin_auth_accepts_active_database_admin_without_claim(monkeypatch):
    admin = SimpleNamespace(is_admin=True, is_active=True)
    monkeypatch.setattr(security, "async_session_maker", lambda: _FakeSession(admin))
    token = security.create_access_token(user_id=9)

    assert await security.get_admin_user_id(token) == 9


def test_admin_decision_rejects_unknown_action():
    with pytest.raises(ValidationError):
        AdminDecisionRequest(action="ESCALATE")


def test_admin_decision_requires_comment_for_rejection():
    with pytest.raises(ValidationError):
        AdminDecisionRequest(action="REJECT", admin_comment="   ")

    assert AdminDecisionRequest(action="REJECT", admin_comment="不符合退款条件").action == "REJECT"


def test_processing_is_a_refund_status():
    assert RefundStatus.PROCESSING.value == "PROCESSING"


def test_access_token_contains_no_role_claim():
    payload = jwt.decode(
        security.create_access_token(user_id=9),
        security.settings.SECRET_KEY,
        algorithms=[security.ALGORITHM],
    )

    assert payload["sub"] == "9"
    assert "is_admin" not in payload


def test_stalled_refund_recovery_is_scheduled():
    from app.celery_app import celery_app

    assert celery_app.conf.beat_schedule["recover-stalled-refunds"]["task"] == "refund.recover_stalled"


def test_admin_client_requires_backend_admin_login(monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "access_token": "server-token",
                "user_id": 3,
                "username": "admin",
                "is_admin": True,
            }

    monkeypatch.setattr("app.frontend.admin_dashboard.requests.post", lambda *a, **k: Response())
    client = AdminClient()

    result = client.login("admin", "secret")

    assert result["success"] is True
    assert client.token == "server-token"
    assert client.admin_id == 3


def test_admin_client_rejects_non_admin_login(monkeypatch):
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "access_token": "user-token",
                "user_id": 1,
                "username": "alice",
                "is_admin": False,
            }

    monkeypatch.setattr("app.frontend.admin_dashboard.requests.post", lambda *a, **k: Response())
    client = AdminClient()

    result = client.login("alice", "secret")

    assert result["success"] is False
    assert client.token is None
    assert client.admin_id is None


# --- A4 卡死恢复核心逻辑测试（P2） ---


class _FakeRecoverResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class _FakeRecoverSession:
    """只模拟 recover_stalled_refunds._recover 需要的最小会话接口。"""

    def __init__(self, refunds, audit_log):
        self._refunds = refunds
        self._audit_log = audit_log
        self._calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, _stmt):
        self._calls += 1
        if self._calls == 1:
            return _FakeRecoverResult(self._refunds)
        return _FakeRecoverResult([self._audit_log])

    def add(self, _obj):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass


def _run_recover(monkeypatch, session, delay_impl):
    monkeypatch.setattr(refund_tasks, "async_session_maker", lambda: session)
    monkeypatch.setattr(
        refund_tasks.DatabaseTask,
        "run_async",
        lambda self, coro: asyncio.run(coro),
    )
    monkeypatch.setattr(refund_tasks.process_refund_payment, "delay", delay_impl)
    return refund_tasks.recover_stalled_refunds.run()


def test_recover_stalled_refunds_reprocesses_and_audits(monkeypatch):
    refund = SimpleNamespace(id=5, status=RefundStatus.PROCESSING, updated_at=None)
    audit_log = SimpleNamespace(refund_application_id=5, decision_metadata={})
    session = _FakeRecoverSession([refund], audit_log)
    delayed = []

    result = _run_recover(monkeypatch, session, lambda refund_id=None: delayed.append(refund_id))

    assert refund.status == RefundStatus.APPROVED
    assert audit_log.decision_metadata["payment_recovered_by"] == "refund.recover_stalled"
    assert delayed == [5]
    assert result["recovered_refund_ids"] == [5]
    assert result["requeue_errors"] == {}


def test_recover_stalled_refunds_captures_requeue_failure(monkeypatch):
    refund = SimpleNamespace(id=5, status=RefundStatus.PROCESSING, updated_at=None)
    audit_log = SimpleNamespace(refund_application_id=5, decision_metadata={})
    session = _FakeRecoverSession([refund], audit_log)

    def failing_delay(refund_id=None):
        raise RuntimeError("broker down")

    result = _run_recover(monkeypatch, session, failing_delay)

    assert refund.status == RefundStatus.APPROVED
    assert result["recovered_refund_ids"] == [5]
    assert result["requeue_errors"]["5"] == "broker down"


def test_recover_stalled_refunds_noop_when_none_stalled(monkeypatch):
    session = _FakeRecoverSession([], None)
    delayed = []

    result = _run_recover(monkeypatch, session, lambda refund_id=None: delayed.append(refund_id))

    assert result["recovered_refund_ids"] == []
    assert result["requeue_errors"] == {}
    assert delayed == []
