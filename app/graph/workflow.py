# app/graph/workflow.py
"""主图：Thin Orchestrator（P3）。

职责（对齐 docs/architecture-update.md 第 3.1 节）：
  * 加载 `ConversationSession`（由 chat.py 在调用前完成）；
  * 读 `active_domain`，按域派发到对应 workflow 子图或单节点；
  * `IntentRouter` 仅在 `active_domain` 为 None / 歧义时调用；
  * 不直接执行退款规则、不查询订单、不生成第二遍话术。

子图节点名固定为：
  - `order_workflow`：OrderWorkflow 子图（P3 新建）
  - `refund_agent`：RefundWorkflow 子图（P1 已落地，P3 改为模块级编译一次复用）
  - `retrieve` / `generate`：POLICY/OTHER 单节点
"""
from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.graph.nodes import generate, intent_router, refund_agent, retrieve
from app.graph.state import AgentState
from app.graph.workflows.order import get_order_subgraph

# 模块加载即编译一次（无 checkpointer，stateless per invoke）。
ORDER_SUBGRAPH = get_order_subgraph()

app_graph = None


# 1. 派发路由：读 active_domain 派发到对应 workflow
def dispatch_router(state: AgentState) -> str:
    """按 active_domain 派发到 OrderWorkflow / retrieve / refund_agent。

    - active_domain 已有 ORDER/POLICY/REFUND：直接派发（避免重复分类）。
    - None 或未识别：进入 intent_router 分类。
    """
    active = state.get("active_domain")
    if active == "ORDER":
        return "order_workflow"
    if active == "POLICY":
        return "retrieve"
    if active == "REFUND":
        return "refund_agent"
    return "classify"


# 2. 意图路由：仅在无 active_domain / 歧义时由 dispatch_router 送入。
def route_intent(state: AgentState) -> str:
    """意图路由（首轮分类后派发）"""
    intent = state.get("intent")
    if intent == "ORDER":
        return "order_workflow"
    elif intent == "POLICY":
        return "retrieve"
    elif intent == "REFUND":
        return "refund_agent"
    return "generate"


# 3. 构建图（只定义结构，不编译）
workflow = StateGraph(AgentState)

# 节点
workflow.add_node("order_workflow", ORDER_SUBGRAPH)
workflow.add_node("retrieve", retrieve)
workflow.add_node("refund_agent", refund_agent)
workflow.add_node("generate", generate)
workflow.add_node("intent_router", intent_router)

# 入口与派发：路由函数不产生状态更新，因此直接作为 START 条件入口。
workflow.add_conditional_edges(
    START,
    dispatch_router,
    {
        "order_workflow": "order_workflow",
        "retrieve": "retrieve",
        "refund_agent": "refund_agent",
        "classify": "intent_router",
    },
)

# 各类终点
workflow.add_edge("order_workflow", END)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("refund_agent", END)

# intent_router → route_intent 派发
workflow.add_conditional_edges(
    "intent_router",
    route_intent,
    {
        "order_workflow": "order_workflow",
        "retrieve": "retrieve",
        "refund_agent": "refund_agent",
        "generate": "generate",
    },
)


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
