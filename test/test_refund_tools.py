# test/test_refund_tools.py
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph.tools import (
    refund_tools,
)
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode


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


async def test_tools():
    """测试退货工具函数"""
    
    print("=" * 60)
    print("🧪 测试 LangGraph Tools")
    print("=" * 60)
    
    user_id = 1  # 假设用户ID为1
    
    # ========== 测试 1: 检查退货资格（不可退商品） ==========
    print("\n📋 测试 1: 检查退货资格 - 运动内衣（应被拒绝）")
    result = await invoke_refund_tool("check_refund_eligibility", {"order_sn": "SN20240001"}, user_id)
    print(result)
    
    # ========== 测试 2: 检查退货资格（可退商品） ==========
    print("\n📋 测试 2: 检查退货资格 - 运动T恤（应该通过）")
    result = await invoke_refund_tool("check_refund_eligibility", {"order_sn": "SN20240003"}, user_id)
    print(result)
    
    # ========== 测试 3: 提交退货申请 ==========
    print("\n📋 测试 3: 提交退货申请 - 篮球鞋")
    result = await invoke_refund_tool("submit_refund_application", {
        "order_sn":  "SN20240004",
        "reason_detail": "鞋码偏大，穿着不舒服",
        "reason_category": "SIZE_NOT_FIT"
    }, user_id)
    print(result)
    
    # ========== 测试 4: 查询所有退货申请 ==========
    print("\n📋 测试 4: 查询所有退货申请")
    result = await invoke_refund_tool("query_refund_status", {}, user_id)
    print(result)
    
    # ========== 测试 5: 查询指定申请 ==========
    print("\n📋 测试 5: 查询指定申请（申请编号 #1）")
    result = await invoke_refund_tool("query_refund_status", {"refund_id": 1}, user_id)
    print(result)
    
    # ========== 测试 6: 跨用户访问（安全测试） ==========
    print("\n📋 测试 6: 跨用户访问 - 用户999查询用户1的订单")
    result = await invoke_refund_tool("check_refund_eligibility", {"order_sn": "SN20240003"}, 999)
    print(result)
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)


if __name__ == "__main__": 
    asyncio.run(test_tools())
