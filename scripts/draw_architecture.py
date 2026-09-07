"""绘制 E-commerce Smart Agent 项目架构总览 PNG。

设计: 左侧标签栏 + 6 层主堆栈 (自上而下) + 右侧质量保障侧带。
  ① 交互层      Gradio 用户端 / 管理员工作台
  ② 接口层      FastAPI (auth/chat/status/admin/websocket)
  ③ 智能体层    Thin Orchestrator + 4 层记忆 + 领域子图
  ④ RAG 层      向量检索 / 引用校验 / 知识库
  ⑤ 工具与执行  ToolCapabilityRegistry / 退款服务 / Celery
  ⑥ 数据与基础  PostgreSQL / Redis / 外部 LLM API
  ⑦ 质量保障    Agent 端到端 / RAGAS / 测试 (右侧侧带)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

# ---- 中文字体 ----
for cand in ("Microsoft YaHei", "SimHei", "PingFang SC",
             "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"):
    if cand in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [cand]
        break
plt.rcParams["axes.unicode_minus"] = False

# ---- 画布 ----
FIG_W, FIG_H, DPI = 13.6, 11.2, 160
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_axis_off()

# ---- 区域 ----
MAIN_X0, MAIN_X1 = 0.5, 9.7
BOX_X0, BOX_X1 = 2.30, 9.60          # 组件分布区(避开左侧标签栏)
SIDE_X0, SIDE_X1 = 10.05, 13.15

REGION_TOP, REGION_BOT = 10.20, 0.55
TIERS, GAP = 6, 0.18
tier_h = (REGION_TOP - REGION_BOT - (TIERS - 1) * GAP) / TIERS
step = tier_h + GAP
tier_c = [REGION_TOP - tier_h / 2 - i * step for i in range(TIERS)]

# ---- 配色 ----
TIER_COLORS = {
    1: ("#E8F1FB", "#2E5C9A"), 2: ("#E5F4F1", "#1E7A6F"),
    3: ("#EFEAF8", "#5E3DA0"), 4: ("#FFF1E5", "#B36A1F"),
    5: ("#FBF6E2", "#8A7A1E"), 6: ("#ECEEF1", "#44505E"),
}
TIER_LABELS = {1: "① 交互层", 2: "② 接口层", 3: "③ 智能体层",
               4: "④ RAG 层", 5: "⑤ 工具与执行", 6: "⑥ 数据与基础"}
TIER_ABBR = {1: "UI", 2: "API", 3: "Agent", 4: "RAG", 5: "Tool", 6: "Data"}
SIDE_BG, SIDE_EDGE = "#FBEAEC", "#A03A4A"
SIDE_LABEL = "⑦ 质量保障 · Quality & Evaluation"

BADGE_W = 1.62


# ---- 工具 ----
def tier_band(idx, yc):
    bg, edge = TIER_COLORS[idx]
    ax.add_patch(FancyBboxPatch(
        (MAIN_X0, yc - tier_h / 2), MAIN_X1 - MAIN_X0, tier_h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=1.3, edgecolor=edge, facecolor=bg, alpha=0.55))
    ax.add_patch(FancyBboxPatch(
        (MAIN_X0 + 0.12, yc - tier_h / 2 + 0.10), BADGE_W, tier_h - 0.20,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.4, edgecolor=edge, facecolor=edge))
    ax.text(MAIN_X0 + 0.12 + BADGE_W / 2, yc + 0.10, TIER_LABELS[idx],
            ha="center", va="center", color="white", fontsize=11.6, fontweight="bold")
    ax.text(MAIN_X0 + 0.12 + BADGE_W / 2, yc - 0.22, TIER_ABBR[idx],
            ha="center", va="center", color="white", fontsize=8.2, alpha=0.9)


def place(widths, bx0=BOX_X0, bx1=BOX_X1, gap=0.30):
    n = len(widths)
    total = sum(widths) + gap * (n - 1)
    x = bx0 + (bx1 - bx0 - total) / 2
    centers = []
    for w in widths:
        centers.append((x + w / 2, w))
        x += w + gap
    return centers


def comp(x, y, w, h, title, sub, edge, bg="white", ts=10.5, ss=8.8):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.09",
        linewidth=1.4, edgecolor=edge, facecolor=bg))
    ax.text(x, y + h * 0.14, title, ha="center", va="center",
            color=edge, fontsize=ts, fontweight="bold")
    if sub:
        ax.text(x, y - h * 0.16, sub, ha="center", va="center",
                color="#444", fontsize=ss)


def arrow(xy1, xy2, color="#44505E", lw=1.6, rad=0.0, style="-|>", ms=14):
    ax.add_patch(FancyArrowPatch(
        xy1, xy2, arrowstyle=style, color=color, lw=lw,
        connectionstyle=f"arc3,rad={rad}", mutation_scale=ms))


# ---- 标题 ----
ax.text(FIG_W / 2, FIG_H - 0.30, "E-commerce Smart Agent",
        ha="center", va="center", fontsize=22, fontweight="bold", color="#1F2A44")
ax.text(FIG_W / 2, FIG_H - 0.58,
        "Thin Orchestrator + 领域子图 + 确定性服务   |   LLM 仅做意图/抽取/受限话术，高风险操作下沉后端",
        ha="center", va="center", fontsize=10.8, color="#5A6478", style="italic")

# ---- ① 交互层 ----
i, yc, edge = 1, tier_c[0], TIER_COLORS[1][1]
tier_band(i, yc)
for (cx, w), (title, sub) in zip(place([2.85, 2.85]), [
        ("Gradio 用户端 UI", ":7860  ·  对话 / SSE 流式"),
        ("Gradio 管理员工作台", ":7861  ·  审核 / 风险")]):
    comp(cx, yc, w, tier_h * 0.56, title, sub, edge)

# ---- ② 接口层 ----
i, yc, edge = 2, tier_c[1], TIER_COLORS[2][1]
tier_band(i, yc)
for (cx, w), (title, sub) in zip(place([6.6]), [
        ("FastAPI 网关", "auth(JWT)  ·  chat(SSE)  ·  status  ·  admin  ·  websocket")]):
    comp(cx, yc, w, tier_h * 0.56, title, sub, edge)

# ---- ③ 智能体层 ----
i, yc, edge = 3, tier_c[2], TIER_COLORS[3][1]
tier_band(i, yc)
cent = place([2.35, 1.55, 2.75])
oc, ow = cent[0]; mc, mw = cent[1]; dc, dw = cent[2]
comp(oc, yc, ow, tier_h * 0.62, "Thin Orchestrator", "LangGraph  ·  compile_app_graph", edge)
comp(mc, yc, mw, tier_h * 0.62, "会话与记忆", "4 层记忆", edge)
gh = tier_h * 0.72
ax.add_patch(FancyBboxPatch(
    (dc - dw / 2, yc - gh / 2), dw, gh,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.4, edgecolor=edge, facecolor="white"))
ax.text(dc, yc + gh * 0.32, "领域子图", ha="center", va="center",
        color=edge, fontsize=10.5, fontweight="bold")
ax.text(dc, yc + gh * 0.05, "OrderWorkflow", ha="center", va="center", color="#333", fontsize=9.2)
ax.text(dc, yc - gh * 0.12, "Policy RAG 节点", ha="center", va="center", color="#333", fontsize=9.2)
ax.text(dc, yc - gh * 0.29, "RefundWorkflow (FSM)", ha="center", va="center", color="#333", fontsize=9.2)

# ---- ④ RAG 层 ----
i, yc, edge = 4, tier_c[3], TIER_COLORS[4][1]
tier_band(i, yc)
for (cx, w), (title, sub) in zip(place([2.30, 2.30, 2.20]), [
        ("向量检索 + 重排", "pgvector  ·  权威重排"),
        ("生成 + 引用校验", "PolicyAnswerGuard"),
        ("条款级知识库", "data/*.md  →  ETL")]):
    comp(cx, yc, w, tier_h * 0.60, title, sub, edge)

# ---- ⑤ 工具与执行层 ----
i, yc, edge = 5, tier_c[4], TIER_COLORS[5][1]
tier_band(i, yc)
for (cx, w), (title, sub) in zip(place([2.30, 2.30, 2.20]), [
        ("ToolCapabilityRegistry", "5 重 Guard"),
        ("退款确定性服务", "资格 / 幂等 / FSM"),
        ("Celery 异步", "支付 / 通知 / 恢复")]):
    comp(cx, yc, w, tier_h * 0.60, title, sub, edge)

# ---- ⑥ 数据与基础设施 ----
i, yc, edge = 6, tier_c[5], TIER_COLORS[6][1]
tier_band(i, yc)
for (cx, w), (title, sub) in zip(place([2.40, 2.20, 2.20]), [
        ("PostgreSQL", "业务表  ·  审计"),
        ("Redis", "Checkpointer + 缓存"),
        ("LLM / Embedding", "外部 (OpenAI-compat)")]):
    comp(cx, yc, w, tier_h * 0.60, title, sub, edge)

# ---- ⑦ 质量保障 侧带 ----
SBC_YC = tier_c[3]
sbh = 3 * tier_h + 2 * GAP
sb_top = SBC_YC + sbh / 2
ax.add_patch(FancyBboxPatch(
    (SIDE_X0, SBC_YC - sbh / 2), SIDE_X1 - SIDE_X0, sbh,
    boxstyle="round,pad=0.02,rounding_size=0.14",
    linewidth=1.3, edgecolor=SIDE_EDGE, facecolor=SIDE_BG, alpha=0.55))
badge_w = SIDE_X1 - SIDE_X0 - 0.22
ax.add_patch(FancyBboxPatch(
    (SIDE_X0 + 0.11, sb_top - 0.10 - 0.42), badge_w, 0.42,
    boxstyle="round,pad=0.02,rounding_size=0.10",
    linewidth=1.4, edgecolor=SIDE_EDGE, facecolor=SIDE_EDGE))
ax.text(SIDE_X0 + 0.11 + badge_w / 2, sb_top - 0.10 - 0.21,
        SIDE_LABEL, ha="center", va="center",
        color="white", fontsize=10.6, fontweight="bold")
q_items = [("Agent 端到端评测", "8 场景 × 7 次  ·  49/49"),
           ("RAG 条款级评测", "RAGAS  ·  45 题条款集"),
           ("单元 + 集成测试", "147 用例  ·  CI")]
item_y = sb_top - 0.10 - 0.42 - 0.38 - 0.36
for title, sub in q_items:
    comp((SIDE_X0 + SIDE_X1) / 2, item_y, SIDE_X1 - SIDE_X0 - 0.36, 0.72,
         title, sub, SIDE_EDGE, ts=10.2, ss=8.6)
    item_y -= 1.05

# ---- 主线纵向箭头 ----
cx = (MAIN_X0 + MAIN_X1) / 2
for i in range(TIERS - 1):
    arrow((cx, tier_c[i] - tier_h / 2 - 0.02),
          (cx, tier_c[i + 1] + tier_h / 2 + 0.02), color="#44505E", lw=1.9)

# ---- 质量保障侧箭头 ----
for ti in (2, 3, 4):
    sy = tier_c[ti]
    arrow((MAIN_X1 + 0.02, sy), (SIDE_X0 - 0.02, sy), color=SIDE_EDGE, lw=1.2)

# ---- 底部脚注 ----
ax.text(0.6, 0.22,
        "阅读方式: 自上而下为主线 (请求 → 编排 → RAG → 工具 → 数据)；右侧质量保障层贯穿评估智能体 / RAG / 工具 / 数据。",
        ha="left", va="center", fontsize=9.2, color="#5A6478")
ax.text(0.6, 0.09,
        "组件命名均对照代码: compile_app_graph (graph/workflow.py) · state_manager (conversation/) · "
        "ToolCapabilityRegistry (graph/tool_registry.py) · PolicyAnswerGuard (services/policy_answer_guard.py)",
        ha="left", va="center", fontsize=8.2, color="#7A8395", style="italic")

# ---- 保存 ----
out = r"D:\PythonProject\CustomerServiceAgent\docs\architecture.png"
fig.savefig(out, dpi=DPI, facecolor="white")
print("saved:", out)
