from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.conversation.state_manager import (
    ConversationStateManager,
    ConversationSummary,
    SlotExtraction,
    default_working_memory,
)


class StubStructuredLLM:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    async def ainvoke(self, _messages):
        if self.error:
            raise self.error
        return self.value


def session(memory=None, summary=None):
    return SimpleNamespace(
        working_memory_json=memory or default_working_memory(),
        conversation_summary=summary,
    )


@pytest.mark.asyncio
async def test_refund_followups_merge_confirmed_slots_without_reasking():
    extractor = StubStructuredLLM(SlotExtraction(domain="REFUND", conversation_goal="申请退款"))
    manager = ConversationStateManager(extractor=extractor, summarizer=StubStructuredLLM())
    first = await manager.prepare_turn(session(), "我要退款")
    assert first["active_domain"] == "REFUND"
    assert first["pending_slots"] == ["order_sn", "refund_reason"]

    extractor.value = SlotExtraction(domain="OTHER")
    second = await manager.prepare_turn(session(first), "SN20240001")
    assert second["active_domain"] == "REFUND"
    assert second["active_order_sn"] == "SN20240001"
    assert second["pending_slots"] == ["refund_reason"]

    extractor.value = SlotExtraction(domain="REFUND", refund_reason="尺码不合适")
    third = await manager.prepare_turn(session(second), "尺码不合适")
    assert third["collected_slots"]["refund_reason"] == "尺码不合适"
    assert third["pending_slots"] == []


@pytest.mark.asyncio
async def test_new_explicit_order_clears_verified_order_id():
    memory = default_working_memory()
    memory.update({"active_domain": "REFUND", "active_order_id": 7, "active_order_sn": "SN1"})
    manager = ConversationStateManager(
        extractor=StubStructuredLLM(SlotExtraction(domain="OTHER")),
        summarizer=StubStructuredLLM(),
    )
    result = await manager.prepare_turn(session(memory), "换成 SN20240002")
    assert result["active_order_id"] is None
    assert result["active_order_sn"] == "SN20240002"


@pytest.mark.asyncio
async def test_summary_validation_preserves_required_business_ids():
    manager = ConversationStateManager(
        extractor=StubStructuredLLM(),
        summarizer=StubStructuredLLM(ConversationSummary(confirmed_facts=["订单 SN20240001"])),
    )
    messages = [HumanMessage(content=str(i)) for i in range(12)]
    summary = await manager._summarize(
        messages,
        {"active_order_id": 9, "active_order_sn": "SN20240001"},
        None,
    )
    assert summary is None


def test_compaction_retains_last_three_complete_turns():
    messages = []
    for i in range(12):
        messages.extend([HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")])
    retained = ConversationStateManager.retained_messages(messages)
    assert [item.content for item in retained] == ["q9", "a9", "q10", "a10", "q11", "a11"]
