# 客服会话历史持久化与重登恢复 — MVP 决策与设计

> 状态：**设计决策 / 待实施**
> 职责边界：本文档只覆盖"用户重登后客服窗口自动恢复最近会话聊天记录"这一功能的决策与实施方案，不重复 `architecture-update.md` 的架构原则与四层记忆模型。实现落地后，相关变更应并回 `architecture-update.md`（设计真源）第 4 节，本文件可归档。
> 范围决策：**不做"新建会话"功能**（现实中一个用户一个客服窗口）。

## 1. 背景与问题

用户期望客服窗口表现与真实场景一致：无论重新登录几次，进入会话后历史聊天记录都在，且能在此基础上继续对话。当前现象是：刷新页面或退出再登录后，对话框里的历史消息被清空。

## 2. 现状分析（基于现有代码）

关键结论：**当前已有会话级业务记忆和 LLM checkpoint 持久化，但只有复用同一 `client_session_id` 或显式 `conversation_id` 时才能连续；真正面向用户的缺口是"气泡可见性"、"登录后拉历史的接口"，以及按账号恢复默认会话的规则。**

| 现状事实 | 代码位置 | 对 MVP 的意义 |
| --- | --- | --- |
| `ConversationSession` 表已存在，含 `working_memory_json` / `conversation_summary` / `active_order_sn` / `last_active_at` / `checkpoint_thread_id` | `app/models/conversation.py:13` | "账号默认会话"无需新建会话表，只改解析规则 |
| `resolve_session` 复用同一行 → `prepare_turn` 每轮 rehydrate 工作记忆 | `app/conversation/state_manager.py:116`、`:169` | 只有复用同一 `client_session_id` 或 `conversation_id` 时，订单号、退款阶段、已收集槽位才连续；新登录会话需按账号恢复 |
| `persist_turn` 落库工作记忆，每次更新 `last_active_at`；`SUMMARY_TRIGGER_TURNS=6` 触发摘要压缩 | `app/conversation/state_manager.py:275` | MVP 的 `ORDER BY last_active_at DESC` 已有现成字段 |
| `resolve_session` 支持 `conversation_id` 主键解析 + `user_id` 越权校验（`Forbidden`） | `app/conversation/state_manager.py:116` | 租户隔离与"按 user_id 查最近会话"可复用 |
| 气泡仅存在于 Gradio 的 `chatbot` 内存 `history`；`handle_logout` 返回 `[]`、`handle_login` 不回填 | `app/frontend/customer_ui.py:390`、`:367` | 刷新/登出即丢 → 真正的缺口 |
| `MessageCard` 表存在，但只存 **B 端卡片**（audit_card / system），按 `thread_id` 索引，C 端对话流未写入 | `app/models/message.py:30`、`app/api/v1/admin.py:247`、`app/tasks/refund_tasks.py:267` | MVP 的 `conversation_messages` **应新建**，不要复用 `MessageCard` |
| assistant 文本仅在**客户端**由 SSE token 拼出；服务端仅在兜底时有 `fallback_answer` | `app/frontend/customer_ui.py:84`、`app/api/v1/chat.py:134` | "保存最终客服回复"需服务端自行拼接（见 §6 修正点②） |

### 2.1 两条独立链路（重要）

- **LLM 记忆连续性** = `ConversationSession.checkpoint_thread_id` + `working_memory_json`。`resolve_session` 命中同一行后才能复用；重新登录生成新的 `client_session_id` 时，当前实现可能创建新行，因此 MVP 必须按账号恢复默认会话。
- **气泡可见性** = 当前只在 Gradio 内存。刷新/登出即丢 → 用户感知到的"信息被清空"。

MVP 真正修的是**气泡回显**，不是"上下文丢失"。验收 1/2 是主战场；3/4/5 在同浏览器下其实已满足。

## 3. 决策结论

1. **做**：新增 `conversation_messages` 表 + `GET /chat/session` 接口，实现重登恢复。
2. **不做"新建会话"**：一个用户 = 一个 append-only 会话。无 `status` 列、无 `new=true` 参数、无新会话按钮。
3. **跨设备自然共享同一会话**：不同设备登同一账号都指向同一行（同一窗口），无需"并发合并"，与"跨设备合并不纳入"不冲突。
4. **会话恢复以 `conversation_id + user_id` 为安全边界**：传入有效 `conversation_id` 时必须校验归属用户，但不能因本次登录生成了新的 `client_session_id` 而拒绝恢复；`client_session_id` 退化为客户端元数据。
5. **账号默认会话查询必须具备并发幂等性**：首次同时打开客服窗口时，只允许最终形成一个可恢复的默认会话。

## 4. 目标设计

### 4.1 数据模型（新增）

```text
conversation_messages
- id               主键（自增）
- conversation_id  FK → conversation_sessions.conversation_id
- user_id          所属用户（冗余索引，便于租户隔离与查询）
- role             user / assistant / system
- content          消息文本（原始文本，不随 checkpoint 压缩而丢失）
- message_type     text / tool / system（MVP 先简化为 text）
- created_at       入库时间
- sequence         （可选）每会话内序号；建议直接用 (conversation_id, created_at, id) 排序，不单独维护
```

> 设计要点：FK 用 `conversation_id`（UUID 主键）而非 `checkpoint_thread_id`；与现有会话模型对齐。`content` 在 transcript 表保留**完整原文**，不受 §6 所述 checkpoint 压缩影响。

### 4.2 接口（新增）

`GET /api/v1/chat/session`

- 鉴权：`get_current_user_id`（复用现有越权防护）。
- 解析：按 `user_id` 查询 `last_active_at DESC LIMIT 1`；无则创建一行。
- 返回：

```json
{
  "conversation_id": "...",
  "messages": [
    {"role": "user", "content": "我想查订单"},
    {"role": "assistant", "content": "请提供订单号"}
  ]
}
```

- 拉历史**限制最近 N 条**（建议 50–100），防长会话 OOM；完整原文仍全量落库，分页留待后续。
- 无历史会话时返回 `{"conversation_id": "...", "messages": []}`。

### 4.3 后端流程

```text
登录
↓ 打开客服页
GET /chat/session        → 按 user_id 取最近会话（无则建）
                        → 返回 conversation_id + 历史消息（限 N 条）
POST /chat              → resolve_session(conversation_id)
                        → prepare_turn()
                        → LangGraph（复用同一 checkpoint_thread_id）
→ 服务端拼接最终回复
→ persist_turn() 保存工作记忆
→ 写入 user + assistant 两条 conversation_messages（失败时写入错误 assistant 消息）
```

### 4.4 前端改动

- `handle_login` 成功后调用 `GET /chat/session`，用 `messages` 回填 `chatbot`，并写 `client.conversation_id`（本地持久化为建议项，见 §6③）。
- 移除"新建会话"按钮相关逻辑（`app/frontend/customer_ui.py:524` `start_new_session` 及其绑定）。
- 后续 `POST /chat` 继续携带 `conversation_id`。

## 5. 验收标准映射

| 验收项 | 当前是否满足 | MVP 是否必要 |
| --- | --- | --- |
| 1. 刷新后消息仍在 | ❌ 不满足（内存 history） | ✅ 必须 |
| 2. 退出再登同一账号，记录自动显示 | ❌ 不满足 | ✅ 必须 |
| 3. 重登后客服理解上一轮上下文 | ✅ 同浏览器已满足 | 仅跨设备时必要（本方案顺带覆盖） |
| 4. `active_order_sn/id` 仍可用 | ✅ 同浏览器已满足 | 仅跨设备时必要（本方案顺带覆盖） |
| 5. 退款中途重登状态恢复 | ✅ 同浏览器已满足 | 仅跨设备时必要（本方案顺带覆盖） |
| 6. 新账号看不到他人会话 | ✅ `get_current_user_id` + `Forbidden` 已保证 | 复用即可 |
| 7. 新建会话后不显示旧消息 | — | 不做（已删除该功能） |

## 6. 修正点与真缺口（落地时必须处理）

① **服务端必须自行拼接最终回复且只落库一次**：`event_generator`（`app/api/v1/chat.py`）需累加过滤掉内部 LLM tag（`POLICY_GUARD_TAG` / `INTERNAL_LLM_TAG`）后的 SSE token，得到完整答案；若没有用户 token，则使用现有 `fallback_answer`（`chat.py:134`）。若工作流异常，写入明确的 assistant 错误消息。最终 `final_answer` 在 `persist_turn` 后写入 transcript，不能同时重复保存 token 和 fallback。

② **前端落点**：`handle_login` 后调 `GET /chat/session` 回填 history + 写 `conversation_id`。`conversation_id` 本地持久化为建议项；服务端每次仍按 `user_id` 恢复默认会话，不能依赖浏览器保存的 `client_session_id`。

③ **表设计微调**：FK 用 `conversation_id`；`sequence` 可省，排序用 `(conversation_id, created_at, id)`。

④ **transcript 与 checkpoint 并存、各司其职**：transcript 只供 UI 回显；LLM 记忆来自复用的 `checkpoint_thread_id`。`GET /chat/session` 返回的 `conversation_id` 必须能在 `POST /chat` 原样传回，使 `resolve_session` 命中同一行。checkpoint 膨胀已由 `SUMMARY_TRIGGER_TURNS=6` 防线；**transcript 表保留完整原文，不被压缩**。

⑤ **单会话 append-only 无限增长**：拉历史接口按 §4.2 限最近 N 条；完整原文全量落库，分页后续补。
⑥ **恢复时不得强制匹配旧 `client_session_id`**：`conversation_id` 存在时，校验其 `user_id` 归属即可；新的 `client_session_id` 可更新为当前客户端标识或仅作为请求元数据。
⑦ **首次会话创建必须处理竞态**：按账号查不到会话时，创建操作需复用 `IntegrityError` 重查或采用等价数据库唯一约束/事务方案，避免两个设备同时创建两个默认会话。

## 7. 范围外（明确不纳入 MVP）

多历史会话列表、会话搜索、会话标题自动生成、消息分页（仅做限条数截断）、全账号长期偏好记忆、跨会话知识抽取、多设备实时同步、消息编辑/删除、完整工具调用审计展示、新建会话。

## 8. 实施清单（待实施，文件级）

- **模型**：新建 `ConversationMessage`（`app/models/`）。
- **迁移**：新增 Alembic 迁移（本项目为 Python/SQLAlchemy，用 Alembic，非 GreenLoop 的 Flyway）。
- **接口**：`app/api/v1/chat.py` 加 `GET /chat/session`：按 `user_id` 查 `last_active_at DESC LIMIT 1`（无则建）→ 返回 `conversation_id` + 限 N 条 `conversation_messages`。
- **落库**：改造 `chat.py` 的 `event_generator`，在 `persist_turn` 后写入 user + assistant 两条 `ConversationMessage`；服务端统一生成唯一 `final_answer`，异常时写入错误 assistant 消息。
- **解析调整**：`resolve_session` 当 `conversation_id` 为 `None` 时，统一走 `user_id + last_active_at DESC` 取默认行；传入 `conversation_id` 时只校验 `user_id` 归属，不再强制匹配旧 `client_session_id`；`client_session_id` 退化为可丢弃元数据（保留列无妨）。
- **并发创建**：`GET /chat/session` 首次按账号创建会话时，必须具备唯一约束或捕获 `IntegrityError` 后重查的竞态处理。
- **前端**：`app/frontend/customer_ui.py` 的 `handle_login` 调 `GET` 回填；移除 `start_new_session` 及其按钮绑定。
- **测试**：覆盖验收 1/2/6；验证同浏览器重登、不同 `client_session_id` 重登及跨设备登录后气泡回显且 `conversation_id` 稳定；验证 `conversation_id` 恢复不受新 `client_session_id` 阻断；验证 `user_id` 隔离、首次并发创建幂等、assistant token/fallback 只落库一次。
