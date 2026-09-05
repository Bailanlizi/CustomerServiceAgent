"""会话历史结构回归测试。

背景：早期 generate 节点只返回 answer，从不把 AI 回复写回 state["messages"]，
导致会话历史里堆积大量“无人回复的用户消息”。当后续轮次进入退款 Agent（它消费完整
messages）时，模型会把这些历史问题一并重复作答。本文件锁定修复后的两项不变量：

1. 每轮回复后，messages 中 Human/AI 必须成对；
2. 退款 Agent 只看到受控的历史窗口，且窗口起点是用户消息（不拆开 tool 调用配对）。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
# 3. P1 起退款 Agent 委派给 6 阶段 FSM 子图，不再消费历史窗口
# --------------------------------------------------------------------------
# 旧版 refund_agent 是基于 LLM.bind_tools 的 free-form tool loop，需要 LLM
# 看到受控历史窗口避免重复作答旧问题。P1 后退款流程由 FSM 推进，不再依赖
# LLM 决策工具调用，因此本节不再保留历史窗口测试。该测试在新架构下无意义：
# FSM 阶段路由只读 working memory，不读 messages 历史。相关窗口化逻辑
# (select_refund_context_messages / MAX_REFUND_HISTORY_TURNS) 仍保留在
# nodes.py 以防未来回滚，但不再被调用。
@pytest.mark.asyncio
async def test_refund_agent_delegates_to_fsm_subgraph():
    """P1: refund_agent 必须委派给子图，返回 workflow_stage 字段。

    子图在缺 active_order_id 时进入 IDENTIFY_ORDER 阶段，调用 LLM 生成
    追问订单号话术。该测试仅验证委派关系与字段返回，不依赖 LLM 真实调用
    （子图内部会走完整流程，节点会尝试调 LLM；这是端到端 smoke 测试，
    单元测试请看 test_refund_workflow_fsm.py）。
    """
    history = [
        HumanMessage(content="运费怎么算？"),
        AIMessage(content="质量问题由平台承担。"),
        HumanMessage(content="我要退 SN20240001"),
    ]
    # 缺 active_order_id → 进入 IDENTIFY_ORDER，需要 LLM 生成追问话术
    # 这里跳过 LLM 调用的话，子图会因 LLM 不可用而抛错；改用最小可用 state
    # 让子图直接走到 ELIGIBILITY_CHECKED（无活跃订单）路径——但当前会
    # 先尝试 LLM 追问。端到端测试应在 test_refund_workflow_fsm 中用 monkeypatch
    # 隔离 LLM。本测试只验证委派关系：返回 dict 含 workflow_stage。

    # 隔离 LLM 路径：让子图 entry 直接到 DONE（短路）
    # 通过 pre-set refund_submitted_id 让路由跳到 DONE
    result = await nodes.refund_agent({
        "question": "我要退 SN20240001",
        "messages": history,
        "collected_slots": {"refund_submitted_id": 999},
        "active_order_id": 1,
        "active_order_sn": "SN20240001",
        "user_id": 1,
    })
    # 委派关系锁定：refund_agent 返回 dict 且 workflow_stage == "DONE"
    assert isinstance(result, dict)
    assert result.get("workflow_stage") == "DONE"
