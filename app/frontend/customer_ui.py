# app/frontend/customer_ui.py
"""
基于 Gradio 的 C 端用户界面 - v4.0
支持真实登录、多用户切换、横向越权测试
"""
import sys
from pathlib import Path

# 将项目根目录注入 sys.path，使 `import app` 不受启动方式影响
# （用 `python app/frontend/customer_ui.py` 启动时，sys.path[0] 是 app/frontend，
#  顶层 app 包不可见，会导致 No module named 'app'）。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import os
import uuid

import gradio as gr
import requests
from gradio import themes

# 配置
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


class ChatClient:
    """聊天客户端 - 支持真实登录"""

    def __init__(self, token: str, user_id: int, username: str, client_session_id: str):
        self.token = token
        self.user_id = user_id
        self.username = username
        self.client_session_id = client_session_id
        self.conversation_id = None
        # Kept for status compatibility until that API is migrated.
        self.thread_id = None
        # P1: 最近一次 chat 响应的 workflow_stage，由 SSE 流末尾的 stage 字段填充；
        # 前端据此显示「确认提交」按钮。
        self.last_stage: str | None = None
        print(f"✅ 客户端已初始化:  用户={username}, ID={user_id}")

    def load_session(self) -> list[dict[str, str]]:
        response = requests.get(
            f"{API_BASE_URL}/chat/session",
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        self.conversation_id = data["conversation_id"]
        self.thread_id = f"conversation:{self.conversation_id}"
        return data.get("messages", [])
    
    def send_message(self, message: str, user_confirmed: bool = False) -> tuple[bool, str, dict]:
        """发送消息到 Agent。

        user_confirmed: P1 - 用户在 WAITING_CONFIRMATION 阶段点击「确认提交」按钮后，
        把上一轮 AI 话术作为 question 重新提交，并把 user_confirmed=True 写入请求体，
        让 refund FSM 子图进入 SUBMITTED 阶段。
        """
        if not message.strip():
            return False, "消息不能为空", {}

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        try:
            print(f"📤 [{self.username}] 发送消息: {message} (user_confirmed={user_confirmed})")

            response = requests.post(
                f"{API_BASE_URL}/chat",
                headers=headers,
                json={
                    "question": message,
                    "client_session_id": self.client_session_id,
                    "conversation_id": self.conversation_id,
                    "user_confirmed": user_confirmed,
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = f"API 错误 {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f": {error_detail.get('detail', response.text)}"
                except:
                    error_msg += f": {response.text[: 200]}"
                return False, error_msg, {}
            
            # 流式接收
            full_answer = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'token' in data:
                                full_answer += data['token']
                            elif data.get('type') == 'session':
                                self.conversation_id = data['conversation_id']
                                self.thread_id = f"conversation:{self.conversation_id}"
                            elif data.get('type') == 'stage':
                                # P1: SSE 流末尾的 stage 事件，更新 last_stage。
                                self.last_stage = data.get('workflow_stage')
                            elif 'error' in data:
                                return False, f"Agent 错误: {data['error']}", {}
                        except json.JSONDecodeError:
                            pass
            
            # 检查状态
            status_info = self.check_status()
            return True, full_answer, status_info
            
        except Exception as e:
            return False, f"请求失败: {e!s}", {}
    
    def check_status(self) -> dict:
        """检查会话状态"""
        if not self.thread_id:
            return {}
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = requests.get(
                f"{API_BASE_URL}/status/{self.thread_id}",
                headers=headers,
                timeout=10
            )
            return response.json() if response.status_code == 200 else {}
        except: 
            return {}


def login_user(username: str, password: str, client_session_id: str) -> tuple[bool, str, ChatClient | None, str]:
    """
    用户登录
    
    Returns:
        (success, message, client, user_info_text)
    """
    if not username or not password:
        return False, "❌ 请输入用户名和密码", None, ""
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/login",
            json={"username":  username, "password": password},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            client = ChatClient(
                token=data["access_token"],
                user_id=data["user_id"],
                username=data["username"],
                client_session_id=client_session_id,
            )
            
            user_info = f"""
**登录成功！**

- 👤 用户名: {data['username']}
- 🆔 用户ID: {data['user_id']}
- 📛 姓名: {data['full_name']}
- 🛡️ 权限: {'管理员' if data['is_admin'] else '普通用户'}
            """
            
            return True, "✅ 登录成功", client, user_info
        else:
            error = response.json().get('detail', '登录失败')
            return False, f"❌ {error}", None, ""
            
    except Exception as e: 
        return False, f"❌ 登录失败:  {e!s}", None, ""


def create_chat_interface():
    """创建聊天界面 - v4.0 优化版"""
    
    # 1. 定义更现代的主题
    theme = themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size=themes.sizes.radius_lg,
        font=[themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
    ).set(
        body_background_fill="#f8fafc",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
    )

    # 2. 深度定制 CSS
    custom_css = """
    /* 全局布局调整 */
    footer {display: none !important;}
    .gradio-container {max-width: 1200px !important; margin: 0 auto;}
    
    /* 登录页样式 */
    .login-wrapper {
        max-width: 420px; 
        margin: 60px auto; 
        padding: 40px !important; 
        background: white; 
        border-radius: 16px; 
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
    }
    .login-header {text-align: center; margin-bottom: 24px;}
    .login-logo {font-size: 48px; margin-bottom: 10px;}
    
    /* 顶部导航栏 */
    .nav-header {
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        background: white; 
        padding: 12px 24px; 
        border-radius: 12px; 
        border: 1px solid #e2e8f0; 
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .user-pill {
        background: #eff6ff; 
        color: #1e40af; 
        padding: 4px 12px; 
        border-radius: 999px; 
        font-size: 0.85em; 
        font-weight: 600;
        border: 1px solid #dbeafe;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* 状态卡片 */
    .audit-card { padding: 16px; border-radius: 8px; margin-top: 8px; font-size: 0.9em; border-left: 4px solid transparent; }
    .audit-card-pending { background: #fffbeb; border-color: #f59e0b; color: #92400e; }
    .audit-card-approved { background: #f0fdf4; border-color: #22c55e; color: #166534; }
    .audit-card-rejected { background: #fef2f2; border-color: #ef4444; color: #991b1b; }
    
    /* 状态栏微调 */
    .status-badge {
        font-size: 0.75rem; 
        padding: 2px 8px; 
        border-radius: 4px; 
        display: inline-block;
        margin-top: 4px;
    }
    """
    
    with gr.Blocks(title="Smart Agent v4.0", theme=theme, css=custom_css) as demo:
        
        # 状态存储
        client_state = gr.State(None)
        browser_session = gr.BrowserState(str(uuid.uuid4()))
        
        # ====================
        #  登录界面 
        # ====================
        with gr.Group(visible=True) as login_panel:
            with gr.Column(elem_classes="login-wrapper"):
                gr.HTML("""
                <div class="login-header">
                    <div class="login-logo"></div>
                    <h2 style="margin:0; color:#1e293b;">欢迎登录</h2>
                    <p style="color:#64748b; margin-top:4px;">E-commerce Smart Agent v4.0</p>
                </div>
                """)
                
                with gr.Group():
                    username_input = gr.Textbox(
                        label="账号", 
                        placeholder="请输入用户名 (如 alice)", 
                        scale=1
                    )
                    password_input = gr.Textbox(
                        label="密码", 
                        type="password", 
                        placeholder="请输入密码", 
                        scale=1
                    )
                
                login_btn = gr.Button("立即登录", variant="primary", size="lg")
                login_message = gr.Markdown("", elem_id="login-msg")
                
                # 将测试账号信息折叠，保持界面整洁
                with gr.Accordion("假如你是开发者，点击查看测试账号", open=False):
                    gr.Markdown("""
                    | 用户 | 账号/密码 | 拥有订单 |
                    |---|---|---|
                    | **Alice** | `alice` / `alice123` | SN20240001-003 |
                    | **Bob** | `bob` / `bob123` | SN20240004-005 |
                    | **Admin** | `admin` / `admin123` | 管理员 |
                    """)

        # ====================
        #  聊天主界面
        # ====================
        with gr.Group(visible=False) as chat_panel:
            
            # 顶部导航栏
            with gr.Group(elem_classes="nav-header"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4, min_width=200):
                        gr.Markdown("###  智能客服助手", elem_classes="m-0")
                    
                    with gr.Column(scale=4, min_width=200):
                        # 这里用 HTML 动态显示用户信息
                        user_header_display = gr.HTML('<div class="user-pill"> 未登录</div>')
                    
                    with gr.Column(scale=1, min_width=100):
                        logout_btn = gr.Button("退出", size="sm", variant="secondary")

            with gr.Row():
                # 左侧：聊天区 (占宽 75%)
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="对话记录",
                        height=550,
                        placeholder="有什么可以帮您？尝试问问订单状态或退货政策。",
                        avatar_images=("https://ui-avatars.com/api/?name=User&background=random", "https://ui-avatars.com/api/?name=Bot&background=0D8ABC&color=fff"),
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            show_label=False,
                            placeholder="请输入消息... (Enter 发送)",
                            scale=5,
                            container=False,
                            autofocus=True
                        )
                        submit_btn = gr.Button( variant="primary", scale=1, min_width=60)

                    # P1: 退款确认按钮。仅在 last_stage == "WAITING_CONFIRMATION" 时可见；
                    # 用户点击后回传 user_confirmed=True 进入 SUBMITTED 阶段。
                    # visible=gr.update(visible=True) 由 confirm_and_send 内部根据 last_stage 切换。
                    with gr.Row():
                        confirm_btn = gr.Button(
                            "✅ 确认提交退款",
                            variant="primary",
                            visible=False,
                            scale=1,
                        )

                    status_display = gr.HTML("")

                # 右侧：功能面板 (占宽 25%)
                with gr.Column(scale=1):
                    gr.Markdown("###  快捷工具箱")
                    
                    with gr.Accordion(" 订单查询", open=True):
                        btn_query_own = gr.Button("我的订单", size="sm")
                        btn_query_alice = gr.Button("Alice 的订单", size="sm")
                        btn_query_bob = gr.Button("Bob 的订单", size="sm")
                        gr.Markdown("*用于测试越权访问*", elem_classes="text-xs text-gray-400")
                    
                    with gr.Accordion(" 售后服务", open=True):
                        btn_policy = gr.Button("退货政策", size="sm")
                        btn_refund = gr.Button("模拟: 尺码不合退货", size="sm")
                        btn_refund_high = gr.Button("模拟: 大额退款(触发风控)", size="sm")
                    
                    gr.Markdown("---")
                    clear_btn = gr.Button(" 清空显示", variant="stop", size="sm")

        # === 逻辑函数 ===
        
        def handle_login(username, password, stored_session_id):
            session_id = stored_session_id or str(uuid.uuid4())
            success, message, client, _user_info = login_user(username, password, session_id)
            if success:
                try:
                    history = client.load_session()
                except Exception as exc:
                    return (None, gr.update(visible=True), gr.update(visible=False), "", username, password, gr.Warning(f"历史会话加载失败：{exc}"), session_id, [])
                # 提取姓名用于 Header 显示
                name = client.username
                header_html = f'''
                <div style="display:flex; justify-content:flex-end; align-items:center;">
                    <span class="user-pill">👤 {name} (ID: {client.user_id})</span>
                </div>
                '''
                return (
                    client, 
                    gr.update(visible=False), # 隐藏登录
                    gr.update(visible=True),  # 显示聊天
                    header_html,
                    "", "", # 清空输入框
                    gr.Info("登录成功！"), # 使用 Gradio 内置通知
                    session_id,
                    history,
                )
            else:
                return (None, gr.update(visible=True), gr.update(visible=False), "", username, password, gr.Warning(message), session_id, [])

        def handle_logout():
            return (
                None, 
                gr.update(visible=True), 
                gr.update(visible=False), 
                '<div class="user-pill">👤 未登录</div>',
                [], # 清空 Chatbot
                ""
            )

        def render_audit_card_v2(status_info: dict) -> str:
            """优化的审核卡片渲染"""
            status = status_info.get("status", "UNKNOWN")
            data = status_info.get("data", {})
            
            if status == "WAITING_ADMIN":
                return f'''
                <div class="audit-card audit-card-pending">
                    <b>⏳ 触发风控审核</b><br>
                    原因：{data.get("trigger_reason", "未知")}<br>
                    风险等级：{data.get("risk_level", "NORMAL")}
                </div>'''
            elif status == "APPROVED":
                return '<div class="audit-card audit-card-approved"> <b>审核通过</b><br>退款流程已启动</div>'
            elif status == "PROCESSING":
                return '<div class="audit-card audit-card-approved"> <b>退款处理中</b><br>请稍后查询处理结果</div>'
            elif status == "REJECTED":
                return f'<div class="audit-card audit-card-rejected"> <b>审核拒绝</b><br>{data.get("admin_comment", "无理由")}</div>'
            return ""

        def send_and_update_v2(message, history, client):
            """适配 Gradio 4.0 messages 格式的消息处理"""
            if not client:
                gr.Warning("会话已过期，请重新登录")
                yield history, message, "", gr.update(visible=False)
                return

            if not message.strip():
                yield history, message, "", gr.update(visible=False)
                return

            # 立即上屏用户消息
            history.append({"role": "user", "content": message})
            yield history, "", '<span class="status-badge" style="background:#e0f2fe; color:#0369a1;">Thinking...</span>', gr.update(visible=False)

            success, response, status_info = client.send_message(message)

            if not success:
                history.append({"role": "assistant", "content": f"❌ Error: {response}"})
                yield history, "", '<span class="status-badge" style="background:#fee2e2; color:#b91c1c;">Error</span>', gr.update(visible=False)
                return

            # 处理回复内容
            final_content = response
            status = status_info.get("status", "PROCESSING")

            # 追加漂亮的 HTML 卡片
            if status in ["WAITING_ADMIN", "APPROVED", "PROCESSING", "REJECTED"]:
                final_content += render_audit_card_v2(status_info)

            history.append({"role": "assistant", "content": final_content})

            status_text = "Ready"
            status_color = "#dcfce7; color:#15803d" # Green
            if status == "WAITING_ADMIN":
                status_text = "Waiting Audit"
                status_color = "#fef3c7; color:#b45309" # Yellow

            # P1: 根据 last_stage 切换 confirm_btn 可见性。
            # WAITING_CONFIRMATION 时显示按钮；其他阶段（DONE / COLLECT_REASON / 等）隐藏。
            confirm_visible = client.last_stage == "WAITING_CONFIRMATION"
            yield history, "", f'<span class="status-badge" style="background:{status_color};">📡 {status_text}</span>', gr.update(visible=confirm_visible)


        def confirm_and_send(history, client):
            """P1: 用户点击「确认提交」按钮的回传逻辑。

            把上一轮退款确认话术作为 question 重新提交，并把 user_confirmed=True
            写入请求体；后端 refund FSM 子图据此进入 SUBMITTED 阶段。
            """
            if not client:
                gr.Warning("会话已过期，请重新登录")
                yield history, gr.update(visible=False)
                return

            # 用"确认提交"作为提问触发后端再次进入 refund FSM
            message = "确认提交退款申请"
            history.append({"role": "user", "content": message})
            yield history, gr.update(visible=False)  # 立即隐藏按钮

            success, response, status_info = client.send_message(message, user_confirmed=True)

            final_content = response if success else f"❌ Error: {response}"
            status = status_info.get("status", "PROCESSING") if success else "ERROR"
            if success and status in ["WAITING_ADMIN", "APPROVED", "PROCESSING", "REJECTED"]:
                final_content += render_audit_card_v2(status_info)
            history.append({"role": "assistant", "content": final_content})

            # SUBMITTED 之后 stage 通常是 DONE；按钮隐藏
            yield history, gr.update(visible=False)

        # === 绑定事件 ===
        login_btn.click(
            handle_login,
            inputs=[username_input, password_input, browser_session],
            outputs=[client_state, login_panel, chat_panel, user_header_display, username_input, password_input, login_message, browser_session, chatbot]
        )
        
        logout_btn.click(
            handle_logout,
            outputs=[client_state, login_panel, chat_panel, user_header_display, chatbot, login_message]
        )
        
        # 回车提交与按钮提交
        msg_input.submit(
            send_and_update_v2,
            inputs=[msg_input, chatbot, client_state],
            outputs=[chatbot, msg_input, status_display, confirm_btn],
        )
        submit_btn.click(
            send_and_update_v2,
            inputs=[msg_input, chatbot, client_state],
            outputs=[chatbot, msg_input, status_display, confirm_btn],
        )

        # P1: 退款确认按钮
        confirm_btn.click(
            confirm_and_send,
            inputs=[chatbot, client_state],
            outputs=[chatbot, confirm_btn],
        )
        
        clear_btn.click(list, outputs=[chatbot])

        # 快捷按钮逻辑
        btn_query_own.click(lambda: "查询我的订单", outputs=msg_input)
        btn_query_alice.click(lambda: "查询订单 SN20240001", outputs=msg_input)
        btn_query_bob.click(lambda: "查询订单 SN20240004", outputs=msg_input)
        btn_policy.click(lambda: "内衣可以退货吗？", outputs=msg_input)
        btn_refund.click(lambda: "我要退货，订单号 SN20240003，尺码不合适", outputs=msg_input)
        btn_refund_high.click(lambda: "我要退款 2500 元，订单 SN20240003，质量有问题", outputs=msg_input)

    return demo


if __name__ == "__main__":
    print("🚀 启动 E-commerce Smart Agent v4.0 客户端界面...")
    print(f"📡 API 地址: {API_BASE_URL}")
    
    demo = create_chat_interface()
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
