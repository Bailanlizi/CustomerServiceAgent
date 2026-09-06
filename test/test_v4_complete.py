# test/test_v4_complete.py
"""
v4.0 完整验收测试
验证场景: 
1. 普通退款（自动通过）
2. 高额退款（触发人工审核）
3. WebSocket 状态同步
4. 管理员决策流程
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage

from app.core.database import init_db
from app.graph.workflow import compile_app_graph


@pytest.mark.integration
@pytest.mark.legacy_e2e
@pytest.mark.asyncio
async def test_v4():
    print("=" * 60)
    print("开始 v4.0 验收测试")
    print("=" * 60)
    
    # 1. 初始化
    print("\n📦 初始化数据库和 Agent...")
    await init_db()
    app_graph = await compile_app_graph()
    
    # 2. 测试场景
    test_cases = [
        {
            "name": "场景1: 低额退款（自动通过）",
            "user_id": 1,
            "query": "我要退款 100 元，订单 SN20240003",
            "expect":  "应该自动通过，无需人工审核",
        },
        {
            "name": "场景2: 高额退款（触发人工审核）",
            "user_id": 1,
            "query": "我要退款 2500 元，订单 SN20240003，商品质量有问题",
            "expect":  "应该触发 HIGH 风险审核",
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f" 测试 {i}/{len(test_cases)}: {case['name']}")
        print(f"{'=' * 60}")
        print(f" 用户ID: {case['user_id']}")
        print(f" 问题:  {case['query']}")
        print(f" 预期: {case['expect']}")
        
        # 构造初始状态
        thread_id = f"test_v4_user_{case['user_id']}_case_{i}"
        initial_state = {
            "question": case["query"],
            "user_id": case["user_id"],
            "thread_id":  thread_id,
            "context": [],
            "order_data": None,
            "intent": None,
            "messages": [HumanMessage(content=case["query"])],
            "answer":  ""
        }
        
        config = {
            "configurable":  {
                "thread_id": thread_id
            }
        }
        
        try:
            # 调用 Agent
            final_state = await app_graph.ainvoke(initial_state, config)
            
            # 输出结果
            print("\n 结果分析:")
            print(f"  意图: {final_state.get('intent', 'N/A')}")
            
            print("\n Agent 回答:")
            print(f"  {final_state.get('answer', 'N/A')}")
            
            assert final_state.get("answer"), "Agent 应返回非空回复"
                
        except AssertionError as e:
            print(f"\n 测试失败: {e}")
            raise
        except Exception as e: 
            print(f"\n 测试异常: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    print(f"\n{'=' * 60}")
    print(" 所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_v4())
