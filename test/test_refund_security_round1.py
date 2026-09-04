from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.v1.admin import AdminDecisionRequest
from app.core import security
from app.frontend.admin_dashboard import AdminClient
from app.models.refund import RefundStatus


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
    forged_claim_token = security.create_access_token(user_id=7, is_admin=True)

    with pytest.raises(HTTPException) as exc_info:
        await security.get_admin_user_id(forged_claim_token)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_admin_auth_accepts_active_database_admin_without_claim(monkeypatch):
    admin = SimpleNamespace(is_admin=True, is_active=True)
    monkeypatch.setattr(security, "async_session_maker", lambda: _FakeSession(admin))
    token = security.create_access_token(user_id=9, is_admin=False)

    assert await security.get_admin_user_id(token) == 9


def test_admin_decision_rejects_unknown_action():
    with pytest.raises(ValidationError):
        AdminDecisionRequest(action="ESCALATE")


def test_processing_is_a_refund_status():
    assert RefundStatus.PROCESSING.value == "PROCESSING"


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

