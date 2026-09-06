from typing import ClassVar

import httpx
import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def client():
    from app.main import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as value:
        yield value


@pytest.mark.asyncio
async def test_auth_session_and_access_control(client, unique_username):
    payload = {
        "username": unique_username,
        "password": "password123",
        "email": f"{unique_username}@example.com",
        "full_name": "Integration User",
    }
    registered = await client.post("/api/v1/register", json=payload)
    assert registered.status_code == 200, registered.text
    token = registered.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    me = await client.get("/api/v1/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["username"] == unique_username

    session = await client.get("/api/v1/chat/session", headers=headers)
    assert session.status_code == 200, session.text
    body = session.json()
    assert body["conversation_id"]
    assert body["messages"] == []

    assert (await client.get("/api/v1/me")).status_code == 401
    assert (await client.get("/api/v1/admin/tasks", headers=headers)).status_code == 403


@pytest.mark.asyncio
async def test_chat_sse_contract_with_deterministic_graph(client, unique_username, monkeypatch):
    from app.conversation.state_manager import default_working_memory
    from app.api.v1 import chat as chat_module
    from app.graph import workflow

    class FakeGraph:
        async def astream_events(self, initial_state, config, version):
            yield {"event": "on_chain_end", "name": "order_workflow", "data": {"output": {"answer": "订单已查询"}}}

        async def aget_state(self, config):
            class Snapshot:
                values: ClassVar = {"messages": [], "workflow_stage": None}

            return Snapshot()

    monkeypatch.setattr(workflow, "app_graph", FakeGraph())
    async def fake_prepare_turn(*args, **kwargs):
        return default_working_memory()

    monkeypatch.setattr(chat_module.conversation_manager, "prepare_turn", fake_prepare_turn)
    payload = {
        "username": unique_username,
        "password": "password123",
        "email": f"{unique_username}@example.com",
        "full_name": "Integration User",
    }
    token = (await client.post("/api/v1/register", json=payload)).json()["access_token"]
    response = await client.post(
        "/api/v1/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "查询我的订单", "client_session_id": "itest"},
    )
    assert response.status_code == 200
    assert "订单已查询" in response.text
    assert "[DONE]" in response.text
