# app/graph/workflow.py
import redis.asyncio as redis
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.prebuilt import ToolNode
from app.graph.state import AgentState
from app.graph.nodes import retrieve, generate, intent_router, query_order, refund_agent, should_call_refund_tool
from app.graph.tools import refund_tools
from app.core.config import settings


app_graph = None


# 1. 定义路由逻辑
def route_intent(state: AgentState):
    """意图路由"""
    intent = state.get("intent")
    if intent == "ORDER": 
        return "query_order"
    elif intent == "POLICY":
        return "retrieve"
    elif intent == "REFUND":
        return "refund_agent"
    return "generate"


# 2. 构建图 (只定义结构，不编译)
workflow = StateGraph(AgentState)

# 添加所有节点
workflow.add_node("intent_router", intent_router)
workflow.add_node("retrieve", retrieve)
workflow.add_node("query_order", query_order)
workflow.add_node("refund_agent", refund_agent)
workflow.add_node("refund_tools", ToolNode(refund_tools))
workflow.add_node("generate", generate)

# 设置入口
workflow.add_edge(START, "intent_router")

# 意图路由
workflow.add_conditional_edges(
    "intent_router",
    route_intent,
    {
        "query_order": "query_order",
        "retrieve": "retrieve",
        "refund_agent": "refund_agent",
        "generate": "generate"
    }
)

# 订单查询后 -> 生成回复
workflow.add_edge("query_order", "generate")

# 知识检索后 -> 生成回复
workflow.add_edge("retrieve", "generate")

workflow.add_conditional_edges(
    "refund_agent",
    should_call_refund_tool,
    {
        "refund_tools": "refund_tools",
        "done": END,
    }
)
workflow.add_edge("refund_tools", "refund_agent")

# 生成回复后结束
workflow.add_edge("generate", END)


async def compile_app_graph():
    """
    编译 LangGraph，初始化 Redis checkpointer
    """
    print("🔧 Compiling LangGraph with Redis checkpointer...")
    
    # 使用 Redis URL 创建 checkpointer（AsyncRedisSaver 接受 redis_url: str）
    checkpointer = AsyncRedisSaver(redis_url=settings.REDIS_URL)
    
    # 编译图
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    print(" LangGraph compiled successfully!")
    return compiled_graph
