#!/usr/bin/env python3
# scripts/test_v3_complete.py
"""
v3.0 完整验收测试
测试场景:  
1. 订单查询（v2 功能）
2. 政策咨询（v1 功能）
3. 退货申请（v3 新功能）
4. 退货资格拒绝
5. 多轮对话退货流程
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app. graph.workflow import compile_app_graph

from app.core.database import init_db


@pytest.mark.integration
@pytest.mark.legacy_e2e
@pytest.mark.asyncio
async def test_v3():
    print("=" * 60)
    print("🚀 开始 v3.0 验收测试")
    print("=" * 60)
    
    # 1. 初始化
    print("\n📦 初始化数据库和 Agent...")
    await init_db()
    app_graph = await compile_app_graph()
    
    # 2. 测试场景
    test_cases = [
        {
            "name": "场景1: 订单查询（v2 功能回归测试）",
            "user_id": 1,
            "query": "查询订单 SN20240003 的状态",
            "expect":  "应该返回订单详情",
        },
        {
            "name": "场景2: 政策咨询（v1 功能回归测试）",
            "user_id": 1,
            "query": "内衣可以退货吗？",
            "expect": "应该从知识库检索回答",
        },
        {
            "name": "场景3: 退货申请 - 一次性提供完整信息",
            "user_id": 1,
            "query": "我要退货，订单号 SN20240003，尺码太大了",
            "expect": "应该成功提交退货申请",
        },
        {
            "name": "场景4: 退货申请 - 不符合条件（内衣）",
            "user_id":  1,
            "query":  "我要退 SN20240001",
            "expect": "应该拒绝退货申请",
        },
        {
            "name": "场景5: 仅说退货（触发多轮对话）",
            "user_id": 1,
            "query": "我想退货",
            "expect": "应该询问订单号",
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"📋 测试 {i}/{len(test_cases)}: {case['name']}")
        print(f"{'=' * 60}")
        print(f"👤 用户ID: {case['user_id']}")
        print(f"❓ 问题: {case['query']}")
        print(f"🎯 预期: {case['expect']}")
        
        # 构造初始状态
        initial_state = {
            "question": case["query"],
            "user_id": case["user_id"],
            "context": [],
            "order_data": None,
            "intent": None,
            "refund_flow_active": None,
            "refund_order_sn": None,
            "refund_step": None,
            "answer": ""
        }
        
        config = {
            "configurable": {
                "thread_id": f"test_v3_user_{case['user_id']}_case_{i}"
            }
        }
        
        try:
            # 调用 Agent
            final_state = await app_graph.ainvoke(initial_state, config)
            
            # 输出结果
            print("\n📊 结果分析:")
            print(f"  意图:  {final_state.get('intent', 'N/A')}")
            print(f"  退货流程活跃: {final_state.get('refund_flow_active', False)}")
            
            print("\n🤖 Agent 回答:")
            print(f"{final_state.get('answer', 'N/A')}")
            
            # 验证逻辑
            if i == 1:
                assert final_state.get('intent') == 'ORDER', "意图识别错误"
                print("\n✅ 测试通过:  订单查询功能正常")
                
            elif i == 2:
                assert final_state.get('intent') == 'POLICY', "意图识别错误"
                print("\n✅ 测试通过: 政策咨询功能正常")
                
            elif i == 3:
                assert final_state.get('intent') == 'REFUND', "意图识别错误"
                assert '申请编号' in final_state.get('answer', ''), "应该包含申请编号"
                print("\n✅ 测试通过: 退货申请成功")
                
            elif i == 4:
                assert final_state.get('intent') == 'REFUND', "意图识别错误"
                assert '不符合' in final_state.get('answer', '') or '不可退货' in final_state.get('answer', ''), "应该拒绝退货"
                print("\n✅ 测试通过: 正确拒绝不符合条件的退货申请")
                
            elif i == 5:
                assert final_state.get('intent') == 'REFUND', "意图识别错误"
                assert '订单号' in final_state.get('answer', ''), "应该询问订单号"
                print("\n✅ 测试通过: 多轮对话流程启动")
                
        except AssertionError as e:
            print(f"\n 测试失败: {e}")
            raise
        except Exception as e:
            print(f"\n 测试异常: {e}")
            import traceback
            traceback. print_exc()
            raise
    
    print(f"\n{'=' * 60}")
    print("🎉 所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_v3())
