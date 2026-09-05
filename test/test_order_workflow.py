# test/test_order_workflow.py
"""P3: OrderWorkflow 子图测试。

覆盖：
  - `_register_capabilities` per-name 幂等（重复 import 不抛错）
  - `query_order_tool` 归属 ORDER，owner_workflow=OrderWorkflow
  - 子图结构：query_order → compose_answer → END
  - node_query_order 成功路径：order_data 含 id（修复 #3）+ order_sn 等字段
  - node_query_order 失败路径：order_data=None、context 给兜底
  - node_compose_order_answer 合成 answer 含订单号与状态
  - 模块级缓存：get_order_subgraph 每次返回同一实例
"""
from __future__ import annotations

import pytest

from app.graph.state import AgentState
from app.graph.tool_registry import (
    ToolCode,
    ToolOutcome,
    tool_registry,
)
from app.graph.workflows import order as order_module
from app.graph.workflows.order import (
    _outcome_to_order_data,
    build_order_subgraph,
    get_order_subgraph,
    node_compose_order_answer,
    node_query_order,
)

# ============================================================
# A. per-name 幂等注册
# ============================================================

def test_register_capabilities_is_idempotent_per_name():
    """重复调用 _register_capabilities 不得抛错（per-name guard）。"""
    # 两次调用，第二次应直接跳过（name 已存在）
    order_module._register_capabilities()
    order_module._register_capabilities()
    cap = tool_registry.get("query_order_tool")
    assert cap.name == "query_order_tool"


def test_query_order_tool_owner_is_order_workflow():
    """P3 修复：query_order_tool 归属从 OrderQuery 改为 OrderWorkflow。"""
    order_module._register_capabilities()
    cap = tool_registry.get("query_order_tool")
    assert cap.domain == "ORDER"
    assert cap.owner_workflow == "OrderWorkflow"
    assert cap.audit_level == "read"
    assert "*" in cap.allowed_stages  # 只读通配


# ============================================================
# B. outcome.data → order_data 字段映射（修复 #3 + #7）
# ============================================================

def test_outcome_to_order_data_maps_order_id_to_id():
    """核心：outcome.data['order_id'] 必须映射为 order_data['id']，
    否则 persist_turn 读 `id` 固化 active_order_id 会失败。
    """
    raw_data = {
        "order_id": 42,
        "order_sn": "SN20240001",
        "status": "SHIPPED",
        "total_amount": "199.00",
        "tracking_number": "TN123",
        "items": [{"name": "book", "qty": 1}],
    }
    out = _outcome_to_order_data(raw_data)
    assert out["id"] == 42
    assert out["order_sn"] == "SN20240001"
    assert out["status"] == "SHIPPED"
    assert out["total_amount"] == "199.00"
    assert out["tracking_number"] == "TN123"
    assert out["items"] == [{"name": "book", "qty": 1}]
    # 必须可被 'SN...' in str(order_data) 通过（v2 集成测试断言）
    assert "SN20240001" in str(out)


# ============================================================
# C. 子图结构
# ============================================================

def test_order_subgraph_has_query_and_compose_nodes():
    """build_order_subgraph 应产出可执行子图，节点名固定。"""
    subgraph = build_order_subgraph()
    # compiled graph has nodes
    nodes = subgraph.get_graph().nodes
    assert "query_order" in nodes
    assert "compose_answer" in nodes


def test_get_order_subgraph_returns_cached_instance():
    """模块级缓存：get_order_subgraph 每次返回同一对象。"""
    a = get_order_subgraph()
    b = get_order_subgraph()
    assert a is b


# ============================================================
# D. node_query_order 行为（用 monkeypatch mock _executor）
# ============================================================

@pytest.mark.asyncio
async def test_node_query_order_success_writes_order_data(monkeypatch):
    """成功路径：outcome.ok=True → order_data 含 id/order_sn 回写 state。"""

    async def fake_invoke(name, *, state, arguments=None):
        return ToolOutcome(
            ok=True,
            code=ToolCode.SUCCESS,
            message="订单号: SN20240001\n状态: SHIPPED",
            data={
                "order_id": 42,
                "order_sn": "SN20240001",
                "status": "SHIPPED",
                "total_amount": "199.00",
                "tracking_number": "TN123",
                "items": [{"name": "book", "qty": 1}],
            },
        )

    monkeypatch.setattr(order_module._executor, "invoke", fake_invoke)

    state: AgentState = {
        "question": "我的订单",
        "user_id": 7,
        "active_domain": "ORDER",
        "active_order_sn": "SN20240001",
    }
    update = await node_query_order(state)
    assert update["order_data"]["id"] == 42
    assert update["order_data"]["order_sn"] == "SN20240001"
    assert update["order_data"]["status"] == "SHIPPED"
    # context 给到 outcome.message
    assert "SN20240001" in update["context"][0]


@pytest.mark.asyncio
async def test_node_query_order_failure_writes_none(monkeypatch):
    """失败路径：outcome.ok=False → order_data=None、context 兜底。"""

    async def fake_invoke(name, *, state, arguments=None):
        return ToolOutcome(
            ok=False,
            code=ToolCode.NOT_AUTHORIZED,
            message="❌ 未查询到该订单，或您无权访问此订单。",
        )

    monkeypatch.setattr(order_module._executor, "invoke", fake_invoke)

    state: AgentState = {
        "question": "未知订单",
        "user_id": 7,
        "active_domain": "ORDER",
        "active_order_sn": None,
    }
    update = await node_query_order(state)
    assert update["order_data"] is None
    assert "未查到" in update["context"][0]


# ============================================================
# E. node_compose_order_answer 行为
# ============================================================

@pytest.mark.asyncio
async def test_compose_order_answer_includes_order_sn_and_status():
    state: AgentState = {
        "question": "我的订单",
        "user_id": 7,
        "order_data": {
            "id": 42,
            "order_sn": "SN20240001",
            "status": "SHIPPED",
            "total_amount": "199.00",
            "tracking_number": "TN123",
            "items": [{"name": "book", "qty": 1}],
        },
    }
    update = await node_compose_order_answer(state)
    answer = update["answer"]
    assert "SN20240001" in answer
    assert "SHIPPED" in answer
    # 回写 AIMessage
    assert len(update["messages"]) == 1
    from langchain_core.messages import AIMessage
    assert isinstance(update["messages"][0], AIMessage)
    assert update["messages"][0].content == answer


@pytest.mark.asyncio
async def test_compose_order_answer_handles_missing_order_data():
    state: AgentState = {
        "question": "我的订单",
        "user_id": 7,
        "order_data": None,
    }
    update = await node_compose_order_answer(state)
    assert "未查询到" in update["answer"] or "无法" in update["answer"]


# ============================================================
# F. end-to-end 子图 ainvoke
# ============================================================

@pytest.mark.asyncio
async def test_order_subgraph_end_to_end(monkeypatch):
    """子图 ainvoke: query_order → compose_answer → END，返回 answer + order_data。"""

    async def fake_invoke(name, *, state, arguments=None):
        return ToolOutcome(
            ok=True,
            code=ToolCode.SUCCESS,
            message="订单号: SN20240001",
            data={
                "order_id": 42,
                "order_sn": "SN20240001",
                "status": "SHIPPED",
                "total_amount": "199.00",
                "tracking_number": "TN123",
                "items": [],
            },
        )

    monkeypatch.setattr(order_module._executor, "invoke", fake_invoke)

    subgraph = get_order_subgraph()
    state: AgentState = {
        "question": "我的订单",
        "user_id": 7,
        "active_domain": "ORDER",
        "active_order_sn": "SN20240001",
    }
    result = await subgraph.ainvoke(state)
    assert result["order_data"]["id"] == 42
    assert result["order_data"]["order_sn"] == "SN20240001"
    assert "SN20240001" in result["answer"]