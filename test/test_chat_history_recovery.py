from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.v1 import chat as chat_module
from app.api.v1.schemas import ChatRequest, ChatSessionResponse
from app.graph import workflow


@pytest.mark.asyncio
async def test_get_chat_session_returns_owned_recent_transcript(monkeypatch):
    conversation_id = uuid4()
    conversation = SimpleNamespace(conversation_id=conversation_id)
    rows = [
        SimpleNamespace(role="user", content="查询订单"),
        SimpleNamespace(role="assistant", content="请提供订单号"),
    ]

    async def resolve(user_id, client_session_id, conversation_id=None):
        assert user_id == 7
        assert client_session_id == "account:7"
        assert conversation_id is None
        return conversation

    async def get_messages(got_id, user_id, limit=100):
        assert got_id == conversation_id
        assert user_id == 7
        assert limit == 100
        return rows

    monkeypatch.setattr(chat_module.conversation_manager, "resolve_session", resolve)
    monkeypatch.setattr(chat_module.conversation_manager, "get_messages", get_messages)

    result = await chat_module.get_chat_session(current_user_id=7)
    assert isinstance(result, ChatSessionResponse)
    assert result.conversation_id == conversation_id
    assert [item.model_dump() for item in result.messages] == [
        {"role": "user", "content": "查询订单"},
        {"role": "assistant", "content": "请提供订单号"},
    ]


@pytest.mark.asyncio
async def test_chat_session_does_not_expose_other_user_messages(monkeypatch):
    async def resolve(*args, **kwargs):
        raise AssertionError("resolver must enforce user ownership")

    monkeypatch.setattr(chat_module.conversation_manager, "resolve_session", resolve)
    with pytest.raises(AssertionError):
        await chat_module.get_chat_session(current_user_id=99)


@pytest.mark.asyncio
async def test_append_messages_writes_user_and_assistant(monkeypatch):
    added = []

    class FakeDB:
        def add_all(self, values):
            added.extend(values)

        async def commit(self):
            return None

    class FakeFactory:
        def __call__(self):
            return self

        async def __aenter__(self):
            return FakeDB()

        async def __aexit__(self, *args):
            return False

    manager = chat_module.conversation_manager
    monkeypatch.setattr("app.conversation.state_manager.async_session_maker", FakeFactory())
    conversation_id = uuid4()
    await manager.append_messages(conversation_id, 7, "你好", "您好，有什么可以帮您？")
    assert [(item.role, item.content, item.message_type) for item in added] == [
        ("user", "你好", "text"),
        ("assistant", "您好，有什么可以帮您？", "text"),
    ]


@pytest.mark.asyncio
async def test_chat_persists_streamed_answer_once(monkeypatch):
    class Graph:
        async def astream_events(self, *_args, **_kwargs):
            yield {"event": "on_chat_model_stream", "tags": [], "data": {"chunk": SimpleNamespace(content="答复")}}

        async def aget_state(self, _config):
            return SimpleNamespace(values={"messages": [], "workflow_stage": None})

    conversation = SimpleNamespace(conversation_id=uuid4(), checkpoint_thread_id="conversation:test")
    saved = []
    monkeypatch.setattr(workflow, "app_graph", Graph())
    monkeypatch.setattr(chat_module.conversation_manager, "resolve_session", lambda *args, **kwargs: _async_value(conversation))
    monkeypatch.setattr(chat_module.conversation_manager, "prepare_turn", lambda *args, **kwargs: _async_value({}))
    monkeypatch.setattr(chat_module.conversation_manager, "persist_turn", lambda *args, **kwargs: _async_value(False))
    monkeypatch.setattr(chat_module.conversation_manager, "append_messages", lambda *args, **kwargs: _capture(saved, args))
    body = await _read_body(await chat_module.chat(ChatRequest(question="你好"), current_user_id=7))
    assert 'data: {"token": "答复"}' in body
    assert len(saved) == 1
    assert saved[0][-2:] == ("你好", "答复")


async def _async_value(value):
    return value


async def _capture(target, args):
    target.append(args)


async def _read_body(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)
