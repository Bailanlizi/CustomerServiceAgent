# test/test_refund_tools.py
import asyncio
import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph.tools import (
    refund_tools,
)
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.graph import END, START, StateGraph
from typing import TypedDict


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
