# app/graph/workflows/order.py
"""P3: OrderWorkflow 子图。

将原主图扁平节点 `query_order` 迁为独立子图，遵循文档 6.2 "各 workflow 自合成"原则：
- `node_query_order` 经 `GuardedToolExecutor` 调 `core_query_order`，把 outcome.data
  映射为同时满足 `persist_turn`（读 `id`/`order_sn`）与 v2 集成测试（断言 `order_data`）
  的字段结构。
- `node_compose_order_answer` 基于该 order_data 确定性合成 answer，回写 `order_data`
  与 `messages`，**不回主图 generate**。
- 子图模块级缓存编译一次（与 refund.py 同模式）。
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from app.graph.state import AgentState
from app.graph.tool_registry import (
    GuardedToolExecutor,
    ToolCapability,
    tool_registry,
)
from app.graph.tools import core_query_order

# ===========================================
# 共享执行器（与 refund.py 同一模式）
# ===========================================
_executor = GuardedToolExecutor()


# ===========================================
# 工具登记（per-name 幂等，与 refund.py 一致）
# ===========================================

_ORDER_CAPABILITIES: tuple[
    tuple[str, ToolCapability, object], ...
] = (
    (
        "query_order_tool",
        ToolCapability(
            name="query_order_tool",
            domain="ORDER",
            allowed_stages=frozenset({"*"}),  # 订单查询在所有退款阶段均允许（只读）
            required_slots=frozenset({"user_id"}),
            writes_to_conversation=True,
            audit_level="read",
            owner_workflow="OrderWorkflow",
        ),
        core_query_order,
    ),
)


def _register_capabilities() -> None:
    """Per-name 幂等注册 ORDER 能力。"""
    for name, capability, handler in _ORDER_CAPABILITIES:
        if name in tool_registry.names():
            continue
        tool_registry.register(capability, handler)


# ===========================================
# 工具结果 → order_data 字段映射（修复 #3 / #7）
# ===========================================

def _outcome_to_order_data(outcome_data: dict[str, Any]) -> dict[str, Any]:
    """把 core_query_order 返回的 outcome.data 映射为符合 persist_turn 期望的 order_data。

    关键字段：
      - `id`: 由 `order_id` 映射而来，供 `state_manager.persist_turn:288` 读 `id` 固化
        `active_order_id`（这是 ORDER→REFUND 连续会话不丢订单归属的关键）。
      - `order_sn` / `status` / `total_amount` / `tracking_number` / `items`:
        既供自合成 answer 使用，也供 v2 集成测试
        `test_v2_complete.py:108 assert 'SN20240001' in str(order_data)` 验证。
    """
    return {
        "id": outcome_data.get("order_id"),
        "order_sn": outcome_data.get("order_sn"),
        "status": outcome_data.get("status"),
        "total_amount": outcome_data.get("total_amount"),
        "tracking_number": outcome_data.get("tracking_number"),
        "items": outcome_data.get("items") or [],
    }


def _workflow_state_dict(state: AgentState) -> dict[str, Any]:
    """把 AgentState 转换为 Registry 需要的 state dict。

    Guard 的 7 步检查需要 active_domain、workflow_stage、collected_slots、
    last_tool_result、active_order_id、user_id 等。这里只读不写，保持稳定。
    """
    keys = (
        "user_id", "active_order_id", "active_order_sn", "thread_id",
        "conversation_id", "active_domain", "intent", "workflow_stage",
        "collected_slots", "last_tool_result",
    )
    return {k: state.get(k) for k in keys}


# ===========================================
# 节点函数
# ===========================================

async def node_query_order(state: AgentState) -> dict[str, Any]:
    """节点 1：通过 GuardedToolExecutor 调 query_order_tool。

    - outcome.ok=True → 映射 outcome.data 为 order_data（含 `id`）回写 state。
    - outcome.ok=False → order_data=None、context 给一句兜底说明（不抛异常，
      由 SSE fallback answer 兜底输出）。
    """
    outcome = await _executor.invoke(
        "query_order_tool",
        state=_workflow_state_dict(state),
        arguments={"question": state.get("question", "")},
    )
    if outcome.ok:
        order_data = _outcome_to_order_data(outcome.data or {})
        return {
            "order_data": order_data,
            "context": [outcome.message],
        }
    # 失败：保留原行为，给上下文兜底
    return {
        "order_data": None,
        "context": ["用户询问了订单，但数据库中未查到相关记录。"],
    }


async def node_compose_order_answer(state: AgentState) -> dict[str, Any]:
    """节点 2：基于 order_data 确定性合成 answer，回写 messages。

    不调用 LLM（ORDER 子图无 LLM 流，token_sent 始终 False；SSE 由 chat.py
    `on_chain_end` 兜底事件 `order_workflow` 输出 answer）。
    """
    order_data = state.get("order_data")
    if not isinstance(order_data, dict):
        answer = "❌ 未查询到该订单，或您无权访问此订单。"
    else:
        order_sn = order_data.get("order_sn") or "未知"
        status = order_data.get("status") or "未知"
        amount = order_data.get("total_amount") or 0
        tracking = order_data.get("tracking_number") or "暂无"
        items = order_data.get("items") or []
        items_str = ", ".join(
            f"{i.get('name', '?')}(x{i.get('qty', 1)})" for i in items
        ) or "暂无商品明细"
        answer = (
            f"【订单详情】\n"
            f"- 订单号: {order_sn}\n"
            f"- 当前状态: {status}\n"
            f"- 订单金额: {amount} 元\n"
            f"- 收货地址/物流: {tracking}\n"
            f"- 商品明细: {items_str}"
        )
    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ===========================================
# 子图构建
# ===========================================

def build_order_subgraph():
    """构建并编译 order 子图（无 checkpointer，由主图托管）。"""
    _register_capabilities()

    workflow = StateGraph(AgentState)

    workflow.add_node("query_order", node_query_order)
    workflow.add_node("compose_answer", node_compose_order_answer)

    workflow.set_entry_point("query_order")
    workflow.add_edge("query_order", "compose_answer")
    workflow.add_edge("compose_answer", END)

    return workflow.compile()


# 模块加载时编译一次并缓存；workflow.py 与外部测试直接复用。
# 子图无 checkpointer、每次 invoke stateless，缓存安全。
_ORDER_SUBGRAPH = build_order_subgraph()


def get_order_subgraph():
    """返回缓存的 order 子图。"""
    return _ORDER_SUBGRAPH


__all__ = [
    "_register_capabilities",
    "build_order_subgraph",
    "get_order_subgraph",
    "node_compose_order_answer",
    "node_query_order",
]
