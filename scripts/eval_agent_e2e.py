#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agent 端到端评估：任务成功率 + 工具调用准确率 + 链路 Trace。

测量口径（对齐 docs/agent-evaluation.md）：
1. 任务成功率     —— 8 场景 × 可机器判定的验收断言（DB / ToolOutcome.code / 结构化返回）
2. 越权拦截率     —— 场景 6 单独统计，不进成功率分母
3. 工具调用准确率 —— 选择 Precision / Recall、顺序、参数（只对有工具调用的场景）
4. 链路 Trace     —— 节点耗时 / LLM token / 工具耗时

用法：
    uv run python scripts/eval_agent_e2e.py --runs 2 [--scenarios S01,S03]

产出：
    eval/runs/agent_e2e_<timestamp>.json   # 机器复查
    eval/runs/agent_e2e_<timestamp>.md     # 面试展示
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from sqlmodel import delete, select

from app.conversation.state_manager import ConversationStateManager
from app.core.config import settings
from app.core.database import async_session_maker, engine, init_db
from app.graph.tool_registry import ToolCapabilityRegistry, tool_registry
from app.models.audit import AuditLog
from app.models.conversation import ConversationMessage, ConversationSession
from app.models.order import Order
from app.models.refund import RefundApplication
from app.models.user import User

manager = ConversationStateManager()

ALICE = "alice"
BOB = "bob"

# 真实工具注册名（app/graph/tools.py）
TOOL_QUERY_ORDER = "query_order_tool"
TOOL_CHECK_ELIGIBILITY = "check_refund_eligibility"
TOOL_SUBMIT = "submit_refund_application"
TOOL_QUERY_STATUS = "query_refund_status"


# ==========================================================
# 工具链采集：monkey-patch ToolCapabilityRegistry.invoke
# 唯一 hook 点，覆盖全部 4 个工具 + 所有调用路径（read 工具不写 audit_log）
# ==========================================================
_tool_calls: list[dict] = []
_original_invoke = ToolCapabilityRegistry.invoke


async def _wrapped_invoke(self: ToolCapabilityRegistry, name: str, state: dict, **arguments: Any):
    t0 = time.perf_counter()
    outcome = await _original_invoke(self, name, state, **arguments)
    _tool_calls.append({
        "name": name,
        "arguments": arguments,
        "user_id": state.get("user_id"),
        "code": getattr(outcome.code, "value", str(outcome.code)),  # 统一转 str，精确匹配
        "ok": outcome.ok,
        "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
    })
    return outcome


ToolCapabilityRegistry.invoke = _wrapped_invoke  # type: ignore[method-assign]


# ==========================================================
# LLM token 采集：monkey-patch nodes.llm 的 ainvoke/astream
# （refund.py 节点内直调 llm.astream 且不传 config，callback 与 astream_events 都追踪不到）
# ==========================================================
_llm_token_store: list[dict] = []
_patched = False


def install_token_callback() -> None:
    """monkey-patch ChatOpenAI 的 ainvoke/astream，记录每次调用的 token 消耗。"""
    global _patched
    if _patched:
        return
    from langchain_openai import ChatOpenAI

    orig_ainvoke = ChatOpenAI.ainvoke
    orig_astream = ChatOpenAI.astream

    async def _ainvoke(self, *args, **kwargs):
        t0 = time.perf_counter()
        resp = await orig_ainvoke(self, *args, **kwargs)
        dur_ms = round((time.perf_counter() - t0) * 1000, 1)
        usage = getattr(resp, "usage_metadata", None) or {}
        _llm_token_store.append({
            "kind": "ainvoke",
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "duration_ms": dur_ms,
            "token_source": "exact_invoke",
        })
        return resp

    async def _astream(self, *args, **kwargs):
        # 流式调用：在末 chunk 合并 usage_metadata；非末 chunk 不带 usage
        t0 = time.perf_counter()
        usage_collected = None
        async for chunk in orig_astream(self, *args, **kwargs):
            # 流式协议下 usage_metadata 通常出现在最后一个 chunk（若 provider 支持）
            chunk_usage = getattr(chunk, "usage_metadata", None)
            if chunk_usage:
                usage_collected = chunk_usage
            yield chunk
        dur_ms = round((time.perf_counter() - t0) * 1000, 1)
        if usage_collected:
            _llm_token_store.append({
                "kind": "astream",
                "prompt_tokens": usage_collected.get("input_tokens"),
                "completion_tokens": usage_collected.get("output_tokens"),
                "duration_ms": dur_ms,
                "token_source": "exact_stream",
            })
        else:
            # provider 未在流中返回 usage（罕见）—— 标 unknown，下游单独计数
            _llm_token_store.append({
                "kind": "astream",
                "prompt_tokens": None,
                "completion_tokens": None,
                "duration_ms": dur_ms,
                "token_source": "unknown",
            })

    ChatOpenAI.ainvoke = _ainvoke  # type: ignore[method-assign]
    ChatOpenAI.astream = _astream  # type: ignore[method-assign]
    _patched = True


# ==========================================================
# 数据库辅助：清理 + 断言查询
# ==========================================================
async def cleanup_all() -> None:
    """启动时全量清理评测涉及的动态数据，保留 users/orders/knowledge_chunks。"""
    async with async_session_maker() as s:
        await s.exec(delete(ConversationMessage))
        await s.exec(delete(ConversationSession))
        await s.exec(delete(RefundApplication))
        await s.exec(delete(AuditLog))
        await s.commit()


async def cleanup_user_conversation(user: str) -> None:
    """每个场景前清理该用户的会话，保证 resolve_session 新建会话（场景隔离）。"""
    async with async_session_maker() as s:
        u = (await s.exec(select(User).where(User.username == user))).first()
        if u is None:
            return
        sessions = (await s.exec(select(ConversationSession).where(ConversationSession.user_id == u.id))).all()
        for sess in sessions:
            await s.exec(delete(ConversationMessage).where(ConversationMessage.conversation_id == sess.conversation_id))
            await s.delete(sess)
        await s.commit()


async def get_user_id(username: str) -> int:
    async with async_session_maker() as s:
        u = (await s.exec(select(User).where(User.username == username))).first()
        return int(u.id)


async def cleanup_order_refund(order_sn: str) -> None:
    """删除指定订单的退款申请（N 次重复运行间隔离退款数据）。

    同时清空进程内幂等缓存 tool_registry._idempotency_cache：其 key 形如
    refund:{user_id}:{order_id}，含 order_id 无法仅凭 order_sn 反推完整 key，
    故整体清空。否则 N≥2 时 submit 命中缓存返回 REFUND_SUBMITTED 却不再落库，
    导致 DB 无记录、退款场景假失败。
    """
    tool_registry._idempotency_cache.clear()
    async with async_session_maker() as s:
        o = (await s.exec(select(Order).where(Order.order_sn == order_sn))).first()
        if o is not None:
            await s.exec(delete(RefundApplication).where(RefundApplication.order_id == o.id))
            await s.commit()


async def refund_count(order_sn: str) -> int:
    async with async_session_maker() as s:
        o = (await s.exec(select(Order).where(Order.order_sn == order_sn))).first()
        if o is None:
            return 0
        rows = (await s.exec(select(RefundApplication).where(RefundApplication.order_id == o.id))).all()
        return len(rows)


async def refund_status(order_sn: str) -> Optional[str]:
    async with async_session_maker() as s:
        o = (await s.exec(select(Order).where(Order.order_sn == order_sn))).first()
        if o is None:
            return None
        r = (await s.exec(select(RefundApplication).where(RefundApplication.order_id == o.id))).first()
        return str(r.status) if r else None  # use_enum_values=True 时 status 已是 str


# ==========================================================
# 场景与结果数据结构
# ==========================================================
@dataclass
class Turn:
    question: str
    user_confirmed: bool = False  # 模拟前端「确认提交」按钮回传
    relogin: bool = False         # 发送前重新 resolve_session（模拟重登）


@dataclass
class TurnResult:
    conversation_id: str
    answer: str
    state: dict
    tool_calls: list[dict]
    trace: dict


@dataclass
class Scenario:
    id: str
    name: str
    user: str
    turns: list[Turn]
    # 允许的工具链路径（多条路径任一命中即 Recall 通过），空表示「不应调工具」
    allowed_chains: list[list[str]]
    # 该场景是否计入「工具准确率」统计
    has_tools: bool
    # 验收断言：返回 (通过, 详情)
    assert_fn: Callable[[list[TurnResult]], Awaitable[tuple[bool, str]]]
    # 每次 run 前需清理退款申请的订单号（None 表示不清理，S08 依赖 S03 建的申请）
    cleanup_order_sn: Optional[str] = None
    # 断言口径：strict = 经过工具层 code 路径；indirect = 仅凭 DB/答案关键词，
    # 未经过 ToolCapabilityRegistry 的 NOT_AUTHORIZED 等 code 路径
    assertion_type: str = "strict"
    assertion_note: Optional[str] = None


# ==========================================================
# 核心：跑一轮对话（复用 resolve_session → prepare_turn → graph → persist_turn）
# ==========================================================
async def run_turn(
    user_id: int,
    question: str,
    conversation_id: Optional[str] = None,
    user_confirmed: bool = False,
) -> TurnResult:
    from app.graph import workflow as wf

    _llm_token_store.clear()
    client_session_id = f"account:{user_id}"
    conversation = await manager.resolve_session(user_id, client_session_id, conversation_id)
    memory = await manager.prepare_turn(conversation, question)

    if user_confirmed:
        slots = dict(memory.get("collected_slots") or {})
        slots["user_confirmed"] = True
        memory["collected_slots"] = slots
        memory["user_confirmed"] = True

    thread_id = conversation.checkpoint_thread_id
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "question": question,
        "user_id": user_id,
        "thread_id": thread_id,
        "conversation_id": str(conversation.conversation_id),
        "client_session_id": client_session_id,
        "context": [],
        "order_data": None,
        "answer": "",
        "messages": [HumanMessage(content=question)],
        **memory,
    }

    _tool_calls.clear()
    trace: dict[str, Any] = {"nodes": [], "llm_calls": []}
    node_start: dict[str, float] = {}
    # LangGraph/LangChain 内部包装节点，非业务语义节点，从 Trace 中剔除
    # 避免 RunnableSequence（外层）与内部业务节点（如 generate）对同一次 LLM call 双计
    _INTERNAL_NODES = {
        "__start__", "__end__", "LangGraph",
        "RunnableSequence", "RunnableLambda", "RunnableParallel",
        "RunnablePassthrough", "RunnableMap", "RunnableBranch",
        "RunnableWithMessageHistory", "RunnableBinding", "RunnableBinding",
        "RunnableAssign", "RunnableWithFallbacks",
    }

    async for event in wf.app_graph.astream_events(initial_state, config, version="v2"):
        kind = event["event"]
        name = event.get("name")
        rid = event.get("run_id")
        if kind == "on_chain_start" and name and rid:
            if name in _INTERNAL_NODES:
                continue
            node_start[rid] = time.perf_counter()
            trace["nodes"].append({"name": name, "duration_ms": None})
        elif kind == "on_chain_end" and name and rid and rid in node_start:
            dur = round((time.perf_counter() - node_start[rid]) * 1000, 1)
            # 回填同 run_id 的节点耗时
            for n in trace["nodes"]:
                if n["name"] == name and n["duration_ms"] is None:
                    n["duration_ms"] = dur
                    break

    # LLM token 从 callback store 读（覆盖节点内直调 llm.astream + prepare_turn 的 extractor）
    trace["llm_calls"] = list(_llm_token_store)

    snapshot = await wf.app_graph.aget_state(config)
    values = dict(snapshot.values)
    await manager.persist_turn(conversation, values)

    return TurnResult(
        conversation_id=str(conversation.conversation_id),
        answer=values.get("answer", "") or "",
        state=values,
        tool_calls=list(_tool_calls),
        trace=trace,
    )


async def run_scenario(sc: Scenario, user_id: int) -> tuple[list[TurnResult], bool, str]:
    if sc.cleanup_order_sn:
        await cleanup_order_refund(sc.cleanup_order_sn)
    await cleanup_user_conversation(sc.user)
    results: list[TurnResult] = []
    conv_id: Optional[str] = None
    for turn in sc.turns:
        if turn.relogin and conv_id:
            # 重登：仅校验 resolve_session 能按 conversation_id 命中同一行
            conversation = await manager.resolve_session(user_id, f"account:{user_id}", conv_id)
            assert str(conversation.conversation_id) == conv_id, "重登后 conversation_id 不一致"
        r = await run_turn(user_id, turn.question, conv_id, turn.user_confirmed)
        conv_id = r.conversation_id
        results.append(r)
    passed, detail = await sc.assert_fn(results)
    return results, passed, detail


# ==========================================================
# 断言辅助
# ==========================================================
def _final_answer(results: list[TurnResult]) -> str:
    return results[-1].answer if results else ""


def _all_tool_names(results: list[TurnResult]) -> list[str]:
    names: list[str] = []
    for r in results:
        names += [tc["name"] for tc in r.tool_calls]
    return names


def _tool_codes(results: list[TurnResult]) -> list[str]:
    codes: list[str] = []
    for r in results:
        codes += [tc["code"] for tc in r.tool_calls]
    return codes


def _chain_hit(actual: list[str], allowed_chains: list[list[str]]) -> bool:
    """实际工具序列是否命中任一允许路径（顺序敏感，允许中间夹 query_order_tool 前置）。"""
    for chain in allowed_chains:
        # 允许路径内可夹带 query_order_tool 前置，但关键依赖顺序不变
        filtered = [t for t in actual if t != TOOL_QUERY_ORDER]
        chain_filtered = [t for t in chain if t != TOOL_QUERY_ORDER]
        if filtered == chain_filtered:
            return True
        # 前缀匹配（实际链路更长也可接受）
        if len(filtered) >= len(chain_filtered) and filtered[: len(chain_filtered)] == chain_filtered:
            return True
    return False


async def _assert_refund_submitted(sn: str) -> tuple[bool, str]:
    status = await refund_status(sn)
    if status == "PENDING":
        return True, f"{sn} 退款申请 status=PENDING"
    return False, f"{sn} 退款申请 status={status}（期望 PENDING）"


async def _assert_order_answer(rs: list[TurnResult], keyword: str) -> tuple[bool, str]:
    answer = _final_answer(rs)
    if not answer:
        return False, "无最终回答"
    # 不跨用户：回答不应包含 bob 的订单号
    if "SN20240004" in answer or "SN20240005" in answer:
        return False, f"回答泄露他人订单：{answer[:80]}"
    if keyword == "我的订单":
        hit = any(sn in answer for sn in ("SN20240001", "SN20240002", "SN20240003"))
        return (hit, f"回答含本人订单" if hit else f"未查到本人订单：{answer[:80]}")
    hit = "SN20240001" in answer
    return (hit, f"回答含 SN20240001" if hit else f"未含 SN20240001：{answer[:80]}")


async def _assert_ask_for_order_sn(rs: list[TurnResult]) -> tuple[bool, str]:
    names = _all_tool_names(rs)
    answer = _final_answer(rs)
    if TOOL_SUBMIT in names:
        return False, "缺订单号却调用了 submit_refund_application"
    # 期望反问要订单号（答案非空即可，或明确含订单号提示）
    ask = any(k in answer for k in ("订单号", "单号", "哪个订单", "哪一单"))
    return (ask, f"反问要订单号：{answer[:80]}") if ask else (False, f"未反问订单号：{answer[:80]}")


async def _assert_policy_answer(rs: list[TurnResult]) -> tuple[bool, str]:
    answer = _final_answer(rs)
    if not answer:
        return False, "无政策回答"
    fallback = any(k in answer for k in ("暂未查询到", "暂无相关", "无法依据", "抱歉"))
    audit = rs[-1].state.get("policy_answer_audit") or {}
    clause_ids = audit.get("applied_clause_ids") or []
    if clause_ids:
        return True, f"引用条款 {clause_ids}"
    if not fallback:
        return True, f"生成了政策回答：{answer[:80]}"
    return False, f"政策回答为空或 fallback：{answer[:80]}"


async def _assert_unauthorized(rs: list[TurnResult]) -> tuple[bool, str]:
    # 业务：不新增 SN20240005 的退款申请
    cnt = await refund_count("SN20240005")
    codes = _tool_codes(rs)
    answer = _final_answer(rs)
    if cnt > 0:
        return False, f"越权却新增了退款申请（{cnt} 条）"
    if "NOT_AUTHORIZED" in codes:
        return True, "工具返回 NOT_AUTHORIZED，未新增申请"
    if any(k in answer for k in ("无权", "没有权限", "未找到", "未查询到", "不在", "无法", "不是您的")):
        return True, f"拒绝越权（无新增申请）：{answer[:80]}"
    return False, f"未明确拦截越权：codes={codes}, answer={answer[:80]}"


async def _assert_idempotent(rs: list[TurnResult]) -> tuple[bool, str]:
    # 场景 3 已为 SN20240003 建申请；此处再次提交应不新增第二条
    cnt = await refund_count("SN20240003")
    codes = _tool_codes(rs)
    answer = _final_answer(rs)
    if cnt != 1:
        return False, f"SN20240003 退款申请数={cnt}（期望 1）"
    if "ALREADY_EXISTS" in codes or "REFUND_SUBMITTED" in codes:
        return True, f"幂等命中（codes={codes}），DB 仅 1 条"
    if any(k in answer for k in ("已有", "已存在", "无需重复", "申请编号")):
        return True, f"提示已有申请（codes={codes}），DB 仅 1 条"
    return False, f"未体现幂等：codes={codes}, answer={answer[:80]}"


# ==========================================================
# 8 场景定义
# ==========================================================
SCENARIOS: list[Scenario] = [
    Scenario(
        id="S01", name="查询我的订单", user=ALICE,
        turns=[Turn("查我的订单")],
        allowed_chains=[[TOOL_QUERY_ORDER]],
        has_tools=True,
        assert_fn=lambda rs: _assert_order_answer(rs, "我的订单"),
    ),
    Scenario(
        id="S02", name="查询指定订单", user=ALICE,
        turns=[Turn("SN20240001 是什么情况")],
        allowed_chains=[[TOOL_QUERY_ORDER]],
        has_tools=True,
        assert_fn=lambda rs: _assert_order_answer(rs, "SN20240001"),
    ),
    Scenario(
        id="S03", name="正常退款全流程", user=ALICE,
        turns=[
            Turn("我要退 SN20240003，质量有问题"),
            Turn("确认提交", user_confirmed=True),
        ],
        allowed_chains=[
            [TOOL_CHECK_ELIGIBILITY, TOOL_SUBMIT],
            [TOOL_QUERY_ORDER, TOOL_CHECK_ELIGIBILITY, TOOL_SUBMIT],
        ],
        has_tools=True,
        assert_fn=lambda rs: _assert_refund_submitted("SN20240003"),
        cleanup_order_sn="SN20240003",
    ),
    Scenario(
        id="S04", name="缺订单号退款", user=ALICE,
        turns=[Turn("我要退款")],
        allowed_chains=[],  # 不应调工具
        has_tools=True,
        assert_fn=_assert_ask_for_order_sn,
    ),
    Scenario(
        id="S05", name="政策问答", user=ALICE,
        turns=[Turn("运动内衣试穿后还能退货吗")],
        allowed_chains=[],
        has_tools=False,  # 走 retrieve+generate，非工具
        assert_fn=_assert_policy_answer,
    ),
    Scenario(
        id="S06", name="越权退款（跨用户）", user=ALICE,
        turns=[Turn("我要退 SN20240005")],  # SN20240005 属于 bob，与 S07 的 SN20240004 隔离
        allowed_chains=[],
        has_tools=True,
        assert_fn=_assert_unauthorized,
        cleanup_order_sn="SN20240005",
        # 诚实标注：当前越权拒绝走 prepare_turn SQL 反查 + identify_order 节点固定文案
        # （refund.py:138-143），不经 ToolCapabilityRegistry 的 NOT_AUTHORIZED code 路径。
        # 要升级为 strict 断言需 D1 业务改造（让 identify_order 调 query_order_tool 触发 code）。
        assertion_type="indirect",
        assertion_note="未经过 ToolCapabilityRegistry NOT_AUTHORIZED 路径；测的是 prepare_turn SQL 反查 + identify_order 固定文案",
    ),
    Scenario(
        id="S07", name="重登后续退款", user=BOB,
        turns=[
            Turn("我要退 SN20240004，尺码不合适"),
            Turn("确认提交", user_confirmed=True, relogin=True),
        ],
        allowed_chains=[
            [TOOL_CHECK_ELIGIBILITY, TOOL_SUBMIT],
            [TOOL_QUERY_ORDER, TOOL_CHECK_ELIGIBILITY, TOOL_SUBMIT],
        ],
        has_tools=True,
        assert_fn=lambda rs: _assert_refund_submitted("SN20240004"),
        cleanup_order_sn="SN20240004",
    ),
    Scenario(
        id="S08", name="重复提交退款", user=ALICE,
        turns=[Turn("我要退 SN20240003，质量有问题")],
        # 已有申请时 agent 先调 eligibility 查询是合理前置（资格层会返回 REJECTED），
        # 不应判"多余"；幂等真正要测的是 DB 不新增第二条（见 _assert_idempotent）
        allowed_chains=[[TOOL_CHECK_ELIGIBILITY]],
        has_tools=True,
        assert_fn=_assert_idempotent,
    ),
]


# ==========================================================
# 指标汇总
# ==========================================================
def _normal_scenarios() -> list[Scenario]:
    return [s for s in SCENARIOS if s.id != "S06"]


def _security_scenario() -> Scenario:
    return next(s for s in SCENARIOS if s.id == "S06")


def _tool_scenarios() -> list[Scenario]:
    return [s for s in SCENARIOS if s.has_tools and s.id != "S06"]


def _is_degenerate_run(run: dict, sc: Scenario) -> bool:
    """判断是否为退化 run：LLM 调用次数过少或 turns 不完整。

    调试用 run（仅触达 prepare_turn、llm_calls=0）不算入工具指标，避免污染聚合。
    阈值 1：最小场景 S01 单轮也只 1 次 LLM 调用；llm_calls=0 说明没真正进 graph
    （被中断或仅触达 prepare_turn 的 extractor）。
    """
    if run.get("error_trace"):
        return True
    turns = run.get("turns") or []
    if len(turns) != len(sc.turns):
        return True
    llm_n = sum(len(t.get("llm_calls", [])) for t in turns)
    return llm_n < 1


def _compute_tool_metrics_single(run: dict, sc: Scenario) -> tuple[float, float, float]:
    """对单次 run 计算 (precision, recall, order_accuracy)。

    返回 None 表示该 run 在该场景下分母为 0（如 allowed_chains=[] 且无调用），
    由调用方决定是否计入聚合。
    """
    actual = run.get("tool_names", [])
    # Recall：期望工具（allowed_chains 并集）实际调了几个
    expected = set()
    for chain in sc.allowed_chains:
        expected |= set(chain)
    expected = expected - {TOOL_QUERY_ORDER}  # query_order_tool 为可选前置
    rec_d = len(expected)
    rec_n = len(expected & set(actual))
    # Precision：调用了多少「不在任何允许链」里的工具
    allowed_all = set()
    for chain in sc.allowed_chains:
        allowed_all |= set(chain)
    extra = [t for t in actual if t not in allowed_all]
    prec_d = len(actual)
    prec_n = len(actual) - len(extra)
    # 顺序：实际序列是否命中允许链
    order_ok = False
    order_valid = True  # 是否可计算（不计算 allowed_chains=[] 且 actual 非空这种异常情况）
    if sc.allowed_chains:
        order_ok = _chain_hit(actual, sc.allowed_chains)
    elif actual:
        # 不应调工具却调了 → order=0（异常路径）
        order_ok = False
    # else: allowed_chains=[] 且 actual=[] → order=1（正确没调）
    else:
        order_ok = True

    precision = prec_n / prec_d if prec_d else None
    recall = rec_n / rec_d if rec_d else None
    order = 1.0 if order_ok else 0.0
    return precision, recall, order


def _agg(values: list, key: str) -> dict:
    """对一组单次指标聚合为 mean/min/max/n。"""
    valid = [v for v in values if v is not None]
    if not valid:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(sum(valid) / len(valid), 4),
        "min": round(min(valid), 4),
        "max": round(max(valid), 4),
        "n": len(valid),
    }


def compute_tool_metrics(runs_by_id: dict[str, list[dict]]) -> dict:
    """工具选择 Precision/Recall/顺序：跨 N 次有效 run 聚合，输出 mean/min/max。

    退化 run（llm_calls<3 / turns 不完整 / error）被剔除，但仍保留在 runs 产物供复查。
    """
    per_run_metrics: dict[str, list[tuple]] = {}  # sid -> [(p, r, o), ...]
    valid_runs_count: dict[str, int] = {}
    degenerate_count: dict[str, int] = {}

    for s in SCENARIOS:
        if not s.has_tools or s.id == "S06":
            continue
        runs = runs_by_id.get(s.id, [])
        per_run_metrics[s.id] = []
        valid_runs_count[s.id] = 0
        degenerate_count[s.id] = 0
        for run in runs:
            if _is_degenerate_run(run, s):
                degenerate_count[s.id] += 1
                continue
            valid_runs_count[s.id] += 1
            p, r, o = _compute_tool_metrics_single(run, s)
            per_run_metrics[s.id].append((p, r, o))

    prec_vals: list = []
    rec_vals: list = []
    order_vals: list = []
    for sid, lst in per_run_metrics.items():
        for p, r, o in lst:
            if p is not None:
                prec_vals.append(p)
            if r is not None:
                rec_vals.append(r)
            order_vals.append(o)

    return {
        "selection_precision": _agg(prec_vals, "precision"),
        "selection_recall": _agg(rec_vals, "recall"),
        "order_accuracy": _agg(order_vals, "order"),
        "valid_runs_per_scenario": valid_runs_count,
        "degenerate_runs_per_scenario": degenerate_count,
    }


# ==========================================================
# 主流程
# ==========================================================
async def main(runs: int, scenario_filter: Optional[str]) -> None:
    print("== Agent 端到端评估 ==")
    print(f"   LLM: {settings.LLM_MODEL} | {settings.OPENAI_BASE_URL}")
    print("   初始化数据库与图...")

    await init_db()
    await cleanup_all()

    from app.graph import workflow as wf
    wf.app_graph = await wf.compile_app_graph()
    install_token_callback()
    print("   图编译完成")

    user_ids = {ALICE: await get_user_id(ALICE), BOB: await get_user_id(BOB)}

    selected = [s for s in SCENARIOS if (not scenario_filter or s.id in scenario_filter)]

    # 场景 8 依赖场景 3 建的申请，确保顺序
    selected.sort(key=lambda s: s.id)

    all_runs: dict[str, list[dict]] = {}
    for s in selected:
        all_runs[s.id] = []

    # 每场景跑 N 次（场景间依赖靠顺序保证：S03 先于 S08）
    for run_i in range(runs):
        for s in selected:
            print(f"   [run {run_i+1}/{runs}] {s.id} {s.name} ...", end=" ", flush=True)
            try:
                results, passed, detail = await run_scenario(s, user_ids[s.user])
                names = _all_tool_names(results)
                codes = _tool_codes(results)
                all_runs[s.id].append({
                    "passed": passed, "detail": detail,
                    "tool_names": names, "tool_codes": codes,
                    "answer": _final_answer(results),
                    "turns": [
                        {"question": t.question, "conversation_id": r.conversation_id,
                         "tool_calls": r.tool_calls,
                         "node_durations": r.trace.get("nodes", []),
                         "llm_calls": r.trace.get("llm_calls", [])}
                        for t, r in zip(s.turns, results)
                    ],
                })
                print("PASS" if passed else "FAIL", "-", detail)
            except Exception as e:  # noqa: BLE001
                import traceback
                print("ERROR", "-", repr(e))
                all_runs[s.id].append({
                    "passed": False, "detail": f"异常: {repr(e)}",
                    "tool_names": [], "tool_codes": [], "answer": "",
                    "turns": [], "error_trace": traceback.format_exc(),
                })

    # 汇总
    normal = _normal_scenarios()
    security = _security_scenario()
    normal_selected = [s for s in normal if s.id in [x.id for x in selected]]

    summary: dict[str, Any] = {
        "llm_model": settings.LLM_MODEL,
        "runs": runs,
        "normal_success_rate": None,
        "indirect_security_signal_rate": None,  # S06 当前为 indirect 口径
        "security_intercept_rate": None,       # 已弃用，保留字段供兼容；strict 改造后(D1)才启用
        "tool_metrics": None,
        "per_scenario": {},
    }

    # 每个场景：N 次里 pass 的次数
    for s in selected:
        runs_of_s = all_runs[s.id]
        passes = sum(1 for r in runs_of_s if r["passed"])
        entry = {
            "name": s.name, "user": s.user,
            "passes": passes, "total": len(runs_of_s),
            "rate": round(passes / len(runs_of_s), 4) if runs_of_s else None,
        }
        # 非默认断言口径（indirect）显式标注，避免报告读者误以为经过工具层 code 路径
        if s.assertion_type != "strict":
            entry["assertion_type"] = s.assertion_type
        if s.assertion_note:
            entry["assertion_note"] = s.assertion_note
        summary["per_scenario"][s.id] = entry

    normal_passes = sum(summary["per_scenario"][s.id]["passes"] for s in normal_selected)
    normal_total = sum(summary["per_scenario"][s.id]["total"] for s in normal_selected)
    if normal_total:
        summary["normal_success_rate"] = round(normal_passes / normal_total, 4)

    if security.id in [x.id for x in selected]:
        sec = summary["per_scenario"][security.id]
        # 字段名诚实：当前 S06 断言是 indirect，不号称"安全拦截率"
        rate = round(sec["passes"] / sec["total"], 4) if sec["total"] else None
        summary["indirect_security_signal_rate"] = rate
        summary["security_intercept_rate"] = None  # 已弃用，保留字段供老脚本兼容，值为 None

    # 工具指标（跨 N 次有效 run 聚合，剔退化 run）
    tool_runs: dict[str, list[dict]] = {}
    for s in selected:
        if s.has_tools and s.id != "S06":
            tool_runs[s.id] = all_runs[s.id]
    summary["tool_metrics"] = compute_tool_metrics(tool_runs)

    # 节点耗时聚合 + LLM-only 延迟（独立指标，不依赖 astream_events 归因）
    node_durs: dict[str, list[float]] = {}
    llm_durs: list[float] = []
    llm_prompt = llm_completion = 0
    llm_calls_n = 0
    token_source_counts = {"exact_invoke": 0, "exact_stream": 0, "unknown": 0}
    for s in selected:
        for r in all_runs[s.id]:
            for turn in r.get("turns", []):
                for n in turn.get("node_durations", []):
                    if n.get("duration_ms") is not None:
                        node_durs.setdefault(n["name"], []).append(n["duration_ms"])
                for llm in turn.get("llm_calls", []):
                    llm_calls_n += 1
                    if llm.get("prompt_tokens"):
                        llm_prompt += llm["prompt_tokens"]
                    if llm.get("completion_tokens"):
                        llm_completion += llm["completion_tokens"]
                    # LLM-only 延迟：从 patch 内 perf_counter 取，避开 astream_events 归因盲区
                    if llm.get("duration_ms") is not None:
                        llm_durs.append(llm["duration_ms"])
                    src = llm.get("token_source", "unknown")
                    token_source_counts[src] = token_source_counts.get(src, 0) + 1
    summary["trace"] = {
        "node_latency": {k: {"count": len(v), "p50_ms": round(_pct(v, 0.5), 1), "p95_ms": round(_pct(v, 0.95), 1)} for k, v in sorted(node_durs.items())},
        "llm_only_latency": {
            "count": len(llm_durs),
            "p50_ms": round(_pct(llm_durs, 0.5), 1) if llm_durs else 0.0,
            "p95_ms": round(_pct(llm_durs, 0.95), 1) if llm_durs else 0.0,
        },
        "llm_calls": llm_calls_n,
        "total_prompt_tokens": llm_prompt,
        "total_completion_tokens": llm_completion,
        "token_source_counts": token_source_counts,
    }

    # 输出 JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    os.makedirs("eval/runs", exist_ok=True)
    json_path = f"eval/runs/agent_e2e_{ts}.json"
    md_path = f"eval/runs/agent_e2e_{ts}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "runs": all_runs}, f, ensure_ascii=False, indent=2)

    # 输出 Markdown
    md = render_markdown(summary, all_runs)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print("\n== 汇总 ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n产物：\n  {json_path}\n  {md_path}")


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(p * len(s)))
    return s[idx]


def render_markdown(summary: dict, all_runs: dict) -> str:
    lines: list[str] = []
    lines.append("# Agent 端到端评估报告\n")
    lines.append(f"- LLM: `{summary['llm_model']}`")
    lines.append(f"- 每个场景重复运行 {summary['runs']} 次（小样本工程评估，不代表生产置信区间）\n")
    lines.append("## 1. 核心指标\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 正常业务成功率 | {summary['normal_success_rate'] if summary['normal_success_rate'] is not None else '—'} |")
    ind_rate = summary.get("indirect_security_signal_rate")
    lines.append(f"| 间接安全信号率（indirect） | {ind_rate if ind_rate is not None else '—'} |")
    if ind_rate is not None:
        lines.append(f"| > 注：indirect = 未经过工具层 NOT_AUTHORIZED code 路径，仅凭 DB/答案关键词断言 |")
    tm = summary.get("tool_metrics") or {}
    def _fmt_metric(m):
        if not m or m.get("mean") is None:
            return "—"
        return f"{m['mean']} (min {m['min']}, max {m['max']}, n={m['n']})"
    lines.append(f"| 工具选择 Precision | {_fmt_metric(tm.get('selection_precision'))} |")
    lines.append(f"| 工具选择 Recall | {_fmt_metric(tm.get('selection_recall'))} |")
    lines.append(f"| 工具顺序正确率 | {_fmt_metric(tm.get('order_accuracy'))} |")
    lines.append("")
    lines.append("## 2. 场景明细\n")
    lines.append("| 场景 | 结果 | 断言口径 |")
    lines.append("|---|---|---|")
    for sid, s in summary["per_scenario"].items():
        mark = f"{s['passes']}/{s['total']} = {s['rate']}"
        atype = s.get("assertion_type", "strict")
        lines.append(f"| {sid} {s['name']} | {mark} | {atype} |")
    lines.append("")
    lines.append("## 3. 节点耗时 (p50/p95 ms)\n")
    lines.append("> 仅含业务语义节点；LangChain 内部包装（RunnableSequence/Lambda 等）已剔除\n")
    lines.append("| 节点 | count | p50 | p95 |")
    lines.append("|---|---|---|---|")
    for k, v in summary["trace"]["node_latency"].items():
        lines.append(f"| {k} | {v['count']} | {v['p50_ms']} | {v['p95_ms']} |")
    lines.append("")
    lines.append("## 4. LLM-only 延迟\n")
    lines.append("> 独立指标，从 patch 内 perf_counter 取，不依赖 astream_events 归因\n")
    ll = summary["trace"].get("llm_only_latency") or {}
    lines.append(f"- LLM 调用 count：{ll.get('count', 0)}")
    lines.append(f"- p50：{ll.get('p50_ms', '—')} ms")
    lines.append(f"- p95：{ll.get('p95_ms', '—')} ms")
    lines.append("")
    lines.append("## 5. LLM Token\n")
    lines.append(f"- LLM 调用次数：{summary['trace']['llm_calls']}")
    lines.append(f"- 总 prompt tokens：{summary['trace']['total_prompt_tokens']}")
    lines.append(f"- 总 completion tokens：{summary['trace']['total_completion_tokens']}")
    tsc = summary["trace"].get("token_source_counts") or {}
    lines.append(f"- token 来源分布：{tsc}（unknown 应为 0；非 0 说明 astream 未返回 usage）")
    lines.append("")
    lines.append("## 6. 失败详情\n")
    for sid, runs_of_s in all_runs.items():
        for r in runs_of_s:
            if not r["passed"]:
                lines.append(f"- `{sid}`：{r['detail']}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 端到端评估")
    parser.add_argument("--runs", type=int, default=2, help="每场景重复次数（默认 2）")
    parser.add_argument("--scenarios", type=str, default=None, help="逗号分隔的场景 id，如 S01,S03")
    args = parser.parse_args()
    scen_filter = set(args.scenarios.split(",")) if args.scenarios else None
    asyncio.run(main(args.runs, scen_filter))
