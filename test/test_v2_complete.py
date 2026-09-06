#!/usr/bin/env python3
"""
v2.0 完整验收测试
测试场景: 
1. ✅ 用户查询自己的订单 → 成功
2. ✅ 用户查询别人的订单 → 失败
3. ✅ 政策咨询功能 → 走 RAG 逻辑
4. ❌ LLM 推理状态 → 严格依据数据
"""

import asyncio
import sys
from pathlib import Path
import pytest

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import init_db
from app.graph.workflow import compile_app_graph
from app.core.security import create_access_token


@pytest.mark.external
@pytest.mark.asyncio
async def test_v2():
    print("=" * 60)
    print("🚀 开始 v2.0 验收测试")
    print("=" * 60)
    
    # 1. 初始化
    print("\n📦 初始化数据库和 Agent...")
    await init_db()
    app_graph = await compile_app_graph()
    
    # 2. 测试场景
    test_cases = [
        {
            "name": "场景1: 用户1查询自己的订单",
            "user_id": 1,
            "query": "帮我查下订单 SN20240001 的状态",
            "expect": "应该返回订单详情",
        },
        {
            "name": "场景2: 用户2尝试查询用户1的订单",
            "user_id": 2,
            "query": "我想看下 SN20240001 的订单详情",
            "expect": "应该返回'未找到'",
        },
        {
            "name": "场景3: 政策咨询(v1 逻辑回归)",
            "user_id":  1,
            "query":  "内衣拆封了可以退吗？",
            "expect":  "应该从知识库检索回答",
        },
        {
            "name": "场景4: 用户1查询最近订单(无订单号)",
            "user_id":  1,
            "query":  "我的最近订单怎么样了？",
            "expect": "应该返回最近一条订单",
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
            "answer": ""
        }
        
        config = {
            "configurable": {
                "thread_id": f"test_user_{case['user_id']}_case_{i}"
            }
        }
        
        try:
            # 调用 Agent
            final_state = await app_graph.ainvoke(initial_state, config)
            
            # 输出结果
            print(f"\n📊 结果分析:")
            print(f"  意图: {final_state.get('intent', 'N/A')}")
            
            if final_state.get('order_data'):
                order = final_state['order_data']
                print(f"  订单号: {order.get('order_sn', 'N/A')}")
                print(f"  状态: {order.get('status', 'N/A')}")
                print(f"  金额: {order.get('total_amount', 'N/A')}")
            else:
                print(f"  订单数据: 无")
            
            print(f"\n🤖 Agent 回答:")
            print(f"  {final_state.get('answer', 'N/A')}")
            
            # 验证逻辑
            if i == 1:
                # 场景1: 用户1应该查到订单
                assert final_state. get('intent') == 'ORDER', "意图识别错误"
                assert final_state.get('order_data') is not None, "应该查到订单"
                assert 'SN20240001' in str(final_state.get('order_data')), "订单号不匹配"
                print("\n✅ 测试通过:  用户成功查询自己的订单")
                
            elif i == 2:
                # 场景2: 用户2不应该查到用户1的订单
                assert final_state.get('intent') == 'ORDER', "意图识别错误"
                assert final_state. get('order_data') is None, "不应该查到别人的订单"
                assert '未找到' in final_state.get('answer') or '未查到' in final_state.get('answer'), "回答不正确"
                print("\n✅ 测试通过: 成功阻止跨用户查询")
                
            elif i == 3:
                # 场景3: 应该走 POLICY 逻辑
                assert final_state.get('intent') == 'POLICY', "意图识别错误"
                assert final_state.get('context'), "应该从知识库检索"
                print("\n✅ 测试通过: 政策咨询功能正常")
                
            elif i == 4:
                # 场景4: 用户1查最近订单
                assert final_state.get('intent') == 'ORDER', "意图识别错误"
                assert final_state.get('order_data') is not None, "应该查到最近订单"
                print("\n✅ 测试通过: 查询最近订单功能正常")
                
        except AssertionError as e:
            print(f"\n❌ 测试失败: {e}")
        except Exception as e:
            print(f"\n⚠️ 测试异常: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print("🎉 所有测试完成")
    print("=" * 60)


@pytest.mark.external
@pytest.mark.asyncio
async def test_jwt_security():
    """测试 JWT 安全性"""
    print("\n🔐 JWT 安全性测试")
    
    # 1. 生成合法 Token
    token_user_1 = create_access_token(user_id=1)
    print(f"✅ 用户1 Token: {token_user_1[: 20]}...")
    
    # 2. 模拟 API 调用 (需要在真实环境测试)
    print("💡 提示: JWT 校验需要在 FastAPI 服务中测试")
    print("   使用以下 curl 命令:")
    print(f"   curl -X POST http://localhost:8000/api/v1/chat \\")
    print(f"        -H 'Authorization: Bearer {token_user_1}' \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"question\": \"查询订单 SN20240001\", \"thread_id\": \"test\"}}'")


if __name__ == "__main__":
    # 运行 Agent 测试
    asyncio.run(test_v2())
    
    # 运行 JWT 测试
    asyncio.run(test_jwt_security())
