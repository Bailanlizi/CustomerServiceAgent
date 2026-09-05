# app/graph/workflow.py
"""主图：意图路由 + 顶层 ToolNode（已接入 Guard）。

P2 architecture:
  * `refund_tools` / `order_tools` 在 `app/graph/tools.py` 中已经全部走
    `GuardedToolExecutor.invoke(...)` 路由到 `ToolCapabilityRegistry`，
    因此顶层 LangChain `ToolNode(refund_tools + order_tools)` 自动满足
    PLAN 1.1 的"不允许任何路径绕过 Guard"要求。
  * `RefundWorkflow` 内部继续通过 `GuardedToolExecutor` 直接调用 core
    handlers，避免 free-form ToolNode 嵌入带来的循环控制问题。
  * `query_order` 节点在主图仍以图节点身份存在（不是 LangChain 工具），
    用于业务流转；其底层 SQL 行为由 `core_query_order` 统一暴露给 Registry，
    未来若要把 `query_order` 也以 LangChain 工具方式暴露给 LLM，可直接
    使用 `app.graph.tools.query_order_tool`。
"""
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.graph.nodes import (
    generate,
    intent_router,
    query_order,
    refund_agent,
    retrieve,
    should_call_refund_tool,
)
from app.graph.state import AgentState
from app.graph.tools import order_tools, refund_tools

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
# 顶层 ToolNode：refund_tools 与 order_tools 内部已通过 GuardedToolExecutor
# 路由到 ToolCapabilityRegistry；因此 LLM 的 tool_call 也会经过 Guard 校验
# （domain / stage / eligibility / user_confirmed / idempotency）。
workflow.add_node("refund_tools", ToolNode(refund_tools + order_tools))
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

    # 必须先建索引：checkpointer 依赖 Redis 的 checkpoint / checkpoint_write 两个
    # 搜索索引读写会话状态。缺少这一步，首次写入会以
    # "Error while searching: No such index checkpoint_write" 失败（包内文档明确要求调用）。
    await checkpointer.asetup()

    # 编译图
    compiled_graph = workflow.compile(checkpointer=checkpointer)

    print(" LangGraph compiled successfully!")
    return compiled_graph