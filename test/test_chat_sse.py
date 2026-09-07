import json

import pytest

from app.api.v1.chat import chat
from app.api.v1.schemas import ChatRequest
from app.graph import workflow


class FakeGraph:
    def __init__(self, events):
        self.events = events

    async def astream_events(self, _state, _config, version):
        assert version == "v2"
        for event in self.events:
            yield event


async def _sse_body(response) -> str:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


def _extract_token_stream(body: str) -> str:
    """把所有 data: {"token": "..."} 事件拼接成完整字符串。

    P2 起 fallback_answer 按切片推送，单测不能再断言整段一次性 token 事件；
    改为校验所有 token 事件拼接后等于完整答案。
    """
    import re
    parts = re.findall(r'data: (\{"token": ".*?"\})', body)
    return "".join(json.loads(p)["token"] for p in parts)


@pytest.mark.asyncio
async def test_policy_sse_sends_verified_chain_output_after_generation(monkeypatch):
    fake_graph = FakeGraph([
        {
            "event": "on_chain_end",
            "name": "generate",
            "data": {"output": {"answer": "这是已完成引用校验的政策答复。"}},
        }
    ])
    monkeypatch.setattr(workflow, "app_graph", fake_graph)

    response = await chat(ChatRequest(question="退货政策是什么？"), current_user_id=1)
    body = await _sse_body(response)

    assert '"type": "session"' in body
    assert '"client_session_id": "default_thread"' in body
    assert _extract_token_stream(body) == "这是已完成引用校验的政策答复。"
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_existing_streaming_path_keeps_streamed_tokens(monkeypatch):
    fake_graph = FakeGraph([
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": type("Chunk", (), {"content": "订单"})()},
        },
        {
            "event": "on_chain_end",
            "name": "generate",
            "data": {"output": {"answer": "订单详情"}},
        },
    ])
    monkeypatch.setattr(workflow, "app_graph", fake_graph)

    response = await chat(ChatRequest(question="查询订单"), current_user_id=1)
    body = await _sse_body(response)

    assert 'data: {"token": "订单"}' in body
    assert "订单详情" not in body
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_policy_guard_tagged_stream_tokens_are_never_forwarded(monkeypatch):
    """带 policy_guard 标签的流式 token（如未校验的 JSON 片段）绝不能输出给前端。"""
    fake_graph = FakeGraph([
        {
            "event": "on_chat_model_stream",
            "tags": ["policy_guard"],
            "data": {"chunk": type("Chunk", (), {"content": '{"answer": "可以退"'})()},
        },
        {
            "event": "on_chat_model_stream",
            "tags": ["policy_guard"],
            "data": {"chunk": type("Chunk", (), {"content": ', "applied_clause_ids": []}'})()},
        },
        {
            "event": "on_chain_end",
            "name": "generate",
            "data": {"output": {"answer": "这是已完成引用校验的政策答复。"}},
        },
    ])
    monkeypatch.setattr(workflow, "app_graph", fake_graph)

    response = await chat(ChatRequest(question="退货政策是什么？"), current_user_id=1)
    body = await _sse_body(response)

    # JSON 片段不得出现在 SSE 输出中
    assert "applied_clause_ids" not in body
    assert '{"answer"' not in body
    # 最终只下发校验通过的完整答案（P2 起按切片推送，校验拼接结果）
    assert _extract_token_stream(body) == "这是已完成引用校验的政策答复。"
    assert body.endswith("data: [DONE]\n\n")
