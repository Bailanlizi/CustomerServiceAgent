"""会话历史结构回归测试。

背景：早期 generate 节点只返回 answer，从不把 AI 回复写回 state["messages"]，
导致会话历史里堆积大量“无人回复的用户消息”。当后续轮次进入退款 Agent（它消费完整
messages）时，模型会把这些历史问题一并重复作答。本文件锁定修复后的两项不变量：

1. 每轮回复后，messages 中 Human/AI 必须成对；
2. 退款 Agent 只看到受控的历史窗口，且窗口起点是用户消息（不拆开 tool 调用配对）。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.graph import nodes
from app.services.policy_answer_guard import PolicyAnswer

EVIDENCE = [
    {
        "content": "质量问题可在签收后 30 天内申请。",
        "source": "03_quality_return_policy.md",
        "clause_ids": ["QUALITY_004"],
        "canonical_clause_ids": [],
        "source_type": "policy",
        "rank": 1,
        "distance": 0.2,
    },
]


class StubChunk:
    """最小化的流式 chunk：支持 `+` 累加与 content/tool_calls 读取。"""

    def __init__(self, content: str):
        self.content = content
        self.tool_calls: list = []

    def __add__(self, other: "StubChunk") -> "StubChunk":
        return StubChunk(self.content + other.content)


class StubLLM:
    """既能直接 astream，也能 bind_tools().astream 的桩件。"""

    def __init__(self, text: str = "这是桩件回答。"):
        self.text = text
        self.seen_messages: list = []

    def bind_tools(self, _tools):
        return self

    async def astream(self, messages):
        self.seen_messages = list(messages)
        for ch in self.text:
            yield StubChunk(ch)


class StubPolicyLLM:
    async def ainvoke(self, messages, config=None):
        return PolicyAnswer(
            answer="质量问题可在签收后 30 天内申请。",
            applied_clause_ids=["QUALITY_004"],
            evidence_clause_ids=["QUALITY_004"],
        )


# --------------------------------------------------------------------------
# 1. 历史窗口选取
# --------------------------------------------------------------------------

def test_select_refund_context_falls_back_to_current_question_when_empty():
    selected = nodes.select_refund_context_messages([], "我要退 SN20240001")

    assert len(selected) == 1
    assert isinstance(selected[0], HumanMessage)
    assert selected[0].content == "我要退 SN20240001"


def test_select_refund_context_keeps_only_recent_turns():
    # 5 轮历史（含本轮），窗口上限为 3 轮
    messages: list = []
    for i in range(1, 6):
        messages.append(HumanMessage(content=f"问题{i}"))
        messages.append(AIMessage(content=f"回答{i}"))

    selected = nodes.select_refund_context_messages(messages, "问题5")

    # 只保留最后 3 轮：Human/AI 各 3 条
    assert len(selected) == 6
    assert selected[0].content == "问题3"
    assert selected[-1].content == "回答5"


def test_select_refund_context_starts_at_human_message():
    """窗口起点必须是 HumanMessage，否则会拆开 tool_calls 与 ToolMessage 的配对。"""
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "check_refund_eligibility",
                "args": {"order_sn": "SN20240001"},
                "id": "call-1",
            }],
        ),
        ToolMessage(content="结果", tool_call_id="call-1"),
        AIMessage(content="已为您预检。"),
        HumanMessage(content="我要退 SN20240001"),
    ]

    selected = nodes.select_refund_context_messages(messages, "我要退 SN20240001")

    assert isinstance(selected[0], HumanMessage)
    # 被丢弃的孤儿 tool 序列不会出现在窗口里
    assert not any(isinstance(m, ToolMessage) for m in selected)


# --------------------------------------------------------------------------
# 2. generate 必须回写 AI 消息
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_writes_ai_message_back_to_history(monkeypatch):
    stub = StubLLM("订单已发货。")
    monkeypatch.setattr(nodes, "llm", stub)

    result = await nodes.generate({
        "question": "我的订单到哪了？",
        "intent": "ORDER",
        "order_data": None,
        "context": [],
        "policy_rules": [],
    })

    assert result["answer"] == "订单已发货。"
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == result["answer"]


@pytest.mark.asyncio
async def test_policy_generate_writes_ai_message_back_to_history(monkeypatch):
    monkeypatch.setattr(nodes, "policy_answer_llm", StubPolicyLLM())

    result = await nodes.generate({
        "question": "质量问题多久可以退？",
        "intent": "POLICY",
        "policy_evidence": EVIDENCE,
        "policy_rules": [],
        "context": [],
    })

    assert result["policy_answer_audit"]["status"] == "passed"
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == result["answer"]


# --------------------------------------------------------------------------
# 3. 退款 Agent 只消费受控窗口
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_refund_agent_receives_only_windowed_history(monkeypatch):
    stub = StubLLM("请问退货原因是什么？")
    monkeypatch.setattr(nodes, "llm", stub)

    history = [
        HumanMessage(content="我的订单到哪了？"),
        AIMessage(content="已发货。"),
        HumanMessage(content="内衣拆封了能退吗？"),
        AIMessage(content="贴身衣物拆封后概不退换。"),
        HumanMessage(content="运费怎么算？"),
        AIMessage(content="质量问题由平台承担。"),
        HumanMessage(content="尺码不合适能退吗？"),
        AIMessage(content="未拆封可退。"),
        HumanMessage(content="我要退 SN20240001"),
    ]

    result = await nodes.refund_agent({
        "question": "我要退 SN20240001",
        "messages": history,
    })

    assert result["answer"] == "请问退货原因是什么？"
    # 首条必须是 SystemMessage，其后只能是最近 3 轮（共 3 条用户消息）
    assert isinstance(stub.seen_messages[0], SystemMessage)
    window = stub.seen_messages[1:]
    assert [m.content for m in window] == [
        "运费怎么算？",
        "质量问题由平台承担。",
        "尺码不合适能退吗？",
        "未拆封可退。",
        "我要退 SN20240001",
    ]
    # 更早的订单/内衣两轮已被裁掉，避免退款 Agent 重复作答旧问题
    assert "内衣拆封了能退吗？" not in [m.content for m in window]
    assert isinstance(window[0], HumanMessage)
