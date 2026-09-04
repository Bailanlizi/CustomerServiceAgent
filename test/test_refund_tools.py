# test/test_refund_tools.py
import asyncio
import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode

from app.graph.tools import (
    check_refund_eligibility,
    refund_tools,
    submit_refund_application,
)


async def invoke_refund_tool(name: str, args: dict, user_id: int) -> str:
    """通过 ToolNode 运行工具，验证身份字段只从 State 注入。"""
    result = await ToolNode(refund_tools).ainvoke({
        "messages": [AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": "test_call", "type": "tool_call"
        }])],
        "user_id": user_id,
        "thread_id": "refund_tool_test",
    })
    return result["messages"][-1].content


def test_toolnode_overrides_model_supplied_identity():
    """模型即使伪造 user_id，ToolNode 也必须以 State 中的用户身份执行。"""
    @tool
    async def identity_probe(
        order_sn: str,
        user_id: Annotated[int, InjectedState("user_id")],
    ) -> str:
        """返回执行时使用的用户 ID。"""
        return f"{order_sn}:{user_id}"

    async def invoke() -> str:
        class ToolTestState(TypedDict):
            messages: list
            user_id: int

        workflow = StateGraph(ToolTestState)
        workflow.add_node("tools", ToolNode([identity_probe]))
        workflow.add_edge(START, "tools")
        workflow.add_edge("tools", END)
        result = await workflow.compile().ainvoke({
            "messages": [AIMessage(content="", tool_calls=[{
                "name": "identity_probe",
                "args": {"order_sn": "SN20240003", "user_id": 999},
                "id": "identity-probe",
                "type": "tool_call",
            }])],
            "user_id": 1,
        })
        return result["messages"][-1].content

    assert asyncio.run(invoke()) == "SN20240003:1"


# ============================================================
# P1 修复回归测试
# - 修复 1：订单号 / 退款原因 / 原因分类由 InjectedState 注入，
#   LLM 即使伪造 args 也无法绕过前置校验。
#
# LangGraph InjectedState 必须通过 ToolNode 在 graph runtime 中调用，单独的
# ToolNode.ainvoke() 会因为 runtime config 缺失而失败。本节测试通过最小 graph
# 包装 ToolNode 来确保 InjectedState 字段被正确注入。
# ============================================================


def _build_tool_graph(tool_fn):
    """构造一个最小 graph，让 ToolNode 能拿到合法的 LangGraph runtime，
    从而把 InjectedState 字段注入到工具调用里。
    注意 LangGraph 的 InjectedState 要求 state schema 显式声明字段。
    """
    from langgraph.graph.message import add_messages

    class S(TypedDict):
        messages: Annotated[list, add_messages]
        user_id: int
        active_order_sn: str | None
        collected_slots: dict
        refund_reason_category: str | None
        thread_id: str

    workflow = StateGraph(S)
    workflow.add_node("tools", ToolNode([tool_fn]))
    workflow.add_edge(START, "tools")
    workflow.add_edge("tools", END)
    return workflow.compile()


@pytest.mark.asyncio
async def test_submit_refund_rejects_when_order_sn_missing_from_state():
    """工作记忆里 active_order_sn 为 None 时，工具必须直接拒绝，
    不能让 LLM 重新声明。"""
    graph = _build_tool_graph(check_refund_eligibility)
    result = await graph.ainvoke(
        {
            "messages": [AIMessage(content="", tool_calls=[{
                "name": "check_refund_eligibility",
                "args": {},
                "id": "call-missing-sn",
                "type": "tool_call",
            }])],
            "user_id": 1,
            "thread_id": "t",
            # LangGraph 的 InjectedState 在 state 缺 key 时直接抛 KeyError；这里
            # 用 None 显式表达"工作记忆里有这个字段但没有值"。
            "active_order_sn": None,
        },
        config={"configurable": {"thread_id": "test-thread"}},
    )
    content = result["messages"][-1].content
    assert "缺少订单号" in content


@pytest.mark.asyncio
async def test_submit_refund_rejects_when_refund_reason_missing_from_state():
    """工作记忆里没有 refund_reason 时，submit 必须直接拒绝；这是修复 1 的关键点。"""
    graph = _build_tool_graph(submit_refund_application)
    # LLM 试图在 args 里塞一个伪造的 reason_detail（来自 prompt 的"建议值"），
    # 但 InjectedState 字段以 working memory 为准，因此必须报错。
    result = await graph.ainvoke(
        {
            "messages": [AIMessage(content="", tool_calls=[{
                "name": "submit_refund_application",
                "args": {"reason_detail": "质量问题"},
                "id": "call-no-reason",
                "type": "tool_call",
            }])],
            "user_id": 1,
            "active_order_sn": "SN20240003",
            "collected_slots": {},
            "refund_reason_category": "QUALITY_ISSUE",
        },
        config={"configurable": {"thread_id": "test-thread-1"}},
    )
    content = result["messages"][-1].content
    assert "缺少退货原因" in content


@pytest.mark.asyncio
async def test_check_refund_eligibility_uses_state_injected_order_sn():
    """check_refund_eligibility 的 order_sn 必须从 InjectedState 读取，
    不能让 LLM 在 args 里塞其他用户的订单号绕过身份校验。"""
    graph = _build_tool_graph(check_refund_eligibility)
    # LLM 伪造 args.order_sn 为 SN20240004（用户2的订单），但 state.active_order_sn
    # 是用户1的 SN20240003；工具应当以 state 为准查询用户1的订单，符合预期。
    result = await graph.ainvoke(
        {
            "messages": [AIMessage(content="", tool_calls=[{
                "name": "check_refund_eligibility",
                "args": {"order_sn": "SN20240004"},  # 用户2的订单号（试图越权）
                "id": "call-spoofed-sn",
                "type": "tool_call",
            }])],
            "user_id": 1,
            "active_order_sn": "SN20240003",  # 工作记忆：用户1的订单
        },
        config={"configurable": {"thread_id": "test-thread-2"}},
    )
    # 应当查询 SN20240003（用户1的订单，SHIPPED 状态，符合退货条件）
    content = result["messages"][-1].content
    assert "SN20240003" in content
    assert "符合退货条件" in content


@pytest.mark.asyncio
async def test_submit_refund_uses_injected_reason_text_in_audit():
    """submit_refund_application 必须从 working memory 取 refund_reason 原文
    与 reason_category 枚举，不能用 collected_slots dict。本测试通过"走到资格校验
    阶段"间接证明两个字段都被正确消费，避免污染数据库。

    若走到提交分支会真创建退款记录，污染数据；这里用 SN20240002 (PENDING) 必然
    被状态规则拒绝，从而在创建前退出，可重复运行。"""
    graph = _build_tool_graph(submit_refund_application)
    result = await graph.ainvoke(
        {
            "messages": [AIMessage(content="", tool_calls=[{
                "name": "submit_refund_application",
                "args": {},
                "id": "call-audit",
            }])],
            "user_id": 1,
            "thread_id": "t-audit",
            "active_order_sn": "SN20240002",
            "collected_slots": {"refund_reason": "尺码不合适"},
            "refund_reason_category": "SIZE_NOT_FIT",
        },
        config={"configurable": {"thread_id": "test-thread-3"}},
    )
    content = result["messages"][-1].content
    # 走到资格校验才能返回"订单状态为 PENDING，只有..."。这证明 order_sn / reason
    # 都从 working memory 正确读取——若任一字段被错误地以空字符串传入工具，会在
    # 前置校验阶段就返回"缺少..."，不会进入资格校验。
    assert "订单状态为 PENDING" in content or "退货申请已提交" in content


async def run_tools_demo():
    """测试退货工具函数"""
    
    print("=" * 60)
    print("🧪 测试 LangGraph Tools")
    print("=" * 60)
    
    user_id = 1  # 假设用户ID为1
    
    # ========== 测试 1: 检查退货资格（不可退商品） ==========
    print("\n📋 测试 1: 检查退货资格 - 运动内衣（应被拒绝）")
    result = await invoke_refund_tool("check_refund_eligibility", {"order_sn": "SN20240001"}, user_id)
    print(result)
    assert "不符合退货条件" in result
    
    # ========== 测试 2: 检查退货资格（可退商品） ==========
    print("\n📋 测试 2: 检查退货资格 - 运动T恤（应该通过）")
    result = await invoke_refund_tool("check_refund_eligibility", {"order_sn": "SN20240003"}, user_id)
    print(result)
    assert "符合退货条件" in result
    
    # ========== 测试 3: 提交退货申请 ==========
    print("\n📋 测试 3: 提交退货申请 - 篮球鞋")
    result = await invoke_refund_tool("submit_refund_application", {
        "order_sn":  "SN20240004",
        "reason_detail": "鞋码偏大，穿着不舒服",
        "reason_category": "SIZE_NOT_FIT"
    }, user_id)
    print(result)
    assert "退货申请" in result
    
    # ========== 测试 4: 查询所有退货申请 ==========
    print("\n📋 测试 4: 查询所有退货申请")
    result = await invoke_refund_tool("query_refund_status", {}, user_id)
    print(result)
    assert "退货申请" in result
    
    # ========== 测试 5: 查询指定申请 ==========
    print("\n📋 测试 5: 查询指定申请（申请编号 #1）")
    result = await invoke_refund_tool("query_refund_status", {"refund_id": 1}, user_id)
    print(result)
    assert "申请" in result
    
    # ========== 测试 6: 跨用户访问（安全测试） ==========
    print("\n📋 测试 6: 跨用户访问 - 用户999查询用户1的订单")
    result = await invoke_refund_tool("check_refund_eligibility", {"order_sn": "SN20240003"}, 999)
    print(result)
    assert "无权访问" in result
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


if __name__ == "__main__": 
    asyncio.run(run_tools_demo())
