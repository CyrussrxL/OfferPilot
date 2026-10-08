"""
Streamlit 前端界面 —— streamlit_app.py

职责：
  提供 OfferPilot 的可视化交互界面，包括：
    - 渐变头部横幅 + 实时数据统计卡片（消息数/事实记忆/情绪/面试场次）
    - 左侧边栏：用户画像编辑、情绪仪表盘、记忆可视化、对话存储管理
    - 右侧主区域三个视图（segmented_control 切换）：
      * 💬 对话：工具栏（日报/清空）、气泡聊天历史、欢迎页示例问题
      * 🎤 模拟面试：按用户画像定制的面试模式卡片 + 面试进度 + 问答历史；
        切入自动开启面试模式，切走自动结束并生成报告记录到画像
      * 🧠 记忆管理：画像/事实记忆管理面板
    - 聊天输入框置于页面顶层：Streamlit 将其固定在页面最底部，
      始终位于所有生成消息的下方

设计理由：
  - 使用 Streamlit 快速构建交互式 Web 界面，适合原型展示。
  - 消息渲染采用 pending 模式：输入框（顶层）只负责追加
    user 消息 + assistant 占位并触发 rerun，工作流执行与流式输出
    都发生在聊天 Tab 内的消息循环中——既保证输入框固定页面底部，
    又保证新消息渲染在正确位置。
  - 对话命名 = 用户ID_时间_第一句话（文件系统安全化），
    加载已有对话后沿用原名保存，避免重复文件。
  - 自定义 CSS 美化：头部横幅、聊天气泡、指标卡片、滚动条。
"""

import json
import os
import re
import sys
import time
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from companion_ai.agents.interview_agent import _infer_interview_topic
from companion_ai.tools.resume_parser import extract_text_from_file, parse_resume
from companion_ai.graph.workflow import (
    run_workflow,
    run_daily_report,
    get_interview_session,
)
from companion_ai.memory.vector_store import vector_store
from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


# ---------------- 视图定义 ----------------

CHAT_VIEW = "💬 对话"
INTERVIEW_VIEW = "🎤 模拟面试"
MEMORY_VIEW = "🧠 记忆管理"
VIEW_OPTIONS = [CHAT_VIEW, INTERVIEW_VIEW, MEMORY_VIEW]


# ---------------- 对话存储 ----------------

def get_conversations_dir():
    """获取对话存储目录。"""
    return os.path.join(os.path.dirname(__file__), "..", "..", "conversations")


def generate_conversation_name(user_id, first_message=""):
    """生成对话名称：用户ID_时间_第一句话。

    第一句话仅保留文件系统安全字符并截断 20 字，
    保证可直接作为 Windows/Linux 文件名。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if first_message:
        snippet = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", first_message).strip()
        snippet = re.sub(r"\s+", " ", snippet)[:20].strip().rstrip(".")
        if snippet:
            return f"{user_id}_{timestamp}_{snippet}"
    return f"{user_id}_{timestamp}"


def load_saved_conversations():
    """从本地文件加载已保存的对话列表。"""
    save_dir = get_conversations_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    conversations = []
    for filename in os.listdir(save_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(save_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conversations.append({
                        "filename": filename,
                        "name": data.get("name", filename),
                        "timestamp": data.get("timestamp", ""),
                        "message_count": len(data.get("chat_history", [])),
                        "user_id": data.get("user_id", ""),
                    })
            except Exception as e:
                logger.error(f"加载对话失败: {e}")

    conversations.sort(key=lambda x: x["timestamp"], reverse=True)
    st.session_state.saved_conversations = conversations


def save_conversation_direct(user_id=None):
    """保存当前对话到本地文件（首次保存时以 用户ID_时间_第一句话 命名）。"""
    if not st.session_state.chat_history:
        return False, None

    save_dir = get_conversations_dir()
    os.makedirs(save_dir, exist_ok=True)

    current_user_id = user_id or st.session_state.user_id

    if st.session_state.current_conversation_name:
        name = st.session_state.current_conversation_name
    else:
        # 首次保存：以对话开始的第一句用户消息命名
        first_user_msg = next(
            (m["content"] for m in st.session_state.chat_history if m["role"] == "user"),
            "",
        )
        name = generate_conversation_name(current_user_id, first_user_msg)
        st.session_state.current_conversation_name = name

    filename = f"{name}.json"
    filepath = os.path.join(save_dir, filename)

    data = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "user_id": current_user_id,
        "chat_history": st.session_state.chat_history,
    }

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        load_saved_conversations()
        return True, name
    except Exception as e:
        logger.error(f"保存失败: {e}")
        return False, None


def load_conversation(filename):
    """从本地文件加载对话（沿用原对话名，后续保存不产生重复文件）。"""
    save_dir = get_conversations_dir()
    filepath = os.path.join(save_dir, filename)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.chat_history = data.get("chat_history", [])
        st.session_state.user_id = data.get("user_id", st.session_state.user_id)
        st.session_state.thread_id = f"thread_{st.session_state.user_id}"
        # 沿用原对话名：加载后继续保存会更新同一文件而非另存新文件
        st.session_state.current_conversation_name = data.get("name")
        st.rerun()
    except Exception as e:
        st.error(f"加载失败: {e}")


def delete_conversation(filename):
    """删除已保存的对话。"""
    save_dir = get_conversations_dir()
    filepath = os.path.join(save_dir, filename)

    try:
        os.remove(filepath)
        load_saved_conversations()
        st.rerun()
    except Exception as e:
        st.error(f"删除失败: {e}")


# ---------------- 会话状态 ----------------

def init_session_state():
    """初始化 Streamlit session state。"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = settings.DEFAULT_USER_ID
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "interview_mode" not in st.session_state:
        st.session_state.interview_mode = False
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"thread_{st.session_state.user_id}"
    if "saved_conversations" not in st.session_state:
        st.session_state.saved_conversations = []
        load_saved_conversations()
    if "streaming_enabled" not in st.session_state:
        st.session_state.streaming_enabled = True
    if "current_conversation_name" not in st.session_state:
        st.session_state.current_conversation_name = None
    if "active_view" not in st.session_state:
        st.session_state.active_view = CHAT_VIEW
    # 定向面试上下文：简历摘要（brief 字符串）/ JD 全文
    if "resume_brief" not in st.session_state:
        st.session_state.resume_brief = ""
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""


def create_new_conversation():
    """创建新对话：保存当前对话（如有），清空历史，对话名待首条消息生成。"""
    if st.session_state.chat_history and st.session_state.current_conversation_name:
        save_conversation_direct()
    st.session_state.chat_history = []
    st.session_state.current_conversation_name = None
    st.rerun()


# ---------------- 样式与展示组件 ----------------

def inject_custom_css():
    """注入自定义 CSS：头部横幅、气泡、指标卡片、滚动条、侧边栏。"""
    st.markdown(
        """
        <style>
        /* 主体背景 */
        .stApp { background: #fafbfe; }
        section.main > div { padding-top: 1.1rem; }

        /* 头部渐变横幅 */
        .app-header {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 55%, #d946ef 100%);
            border-radius: 16px; padding: 1.3rem 1.8rem; color: #fff;
            margin-bottom: 1rem; box-shadow: 0 4px 14px rgba(99, 102, 241, .25);
        }
        .app-header h1 { color: #fff; font-size: 1.85rem; margin: 0 0 .25rem 0; font-weight: 700; }
        .app-header p { color: rgba(255, 255, 255, .9); margin: 0 0 .75rem 0; font-size: .95rem; }
        .feature-badge {
            background: rgba(255, 255, 255, .18); border: 1px solid rgba(255, 255, 255, .35);
            border-radius: 999px; padding: 3px 12px; font-size: .78rem;
            margin-right: 8px; color: #fff; display: inline-block;
        }

        /* 欢迎卡片 */
        .welcome-card {
            background: linear-gradient(135deg, #eef2ff 0%, #fdf2f8 100%);
            border: 1px solid #e0e7ff; border-radius: 16px;
            padding: 1.6rem 1.5rem; text-align: center; margin: 1rem 0 1.4rem 0;
        }
        .welcome-card h3 { margin: 0 0 .4rem 0; color: #312e81; font-size: 1.15rem; }
        .welcome-card p { color: #64748b; margin: 0; font-size: .9rem; }

        /* 聊天气泡 */
        [data-testid="stChatMessage"] {
            border-radius: 14px; border: 1px solid #e5e7eb;
            box-shadow: 0 2px 6px rgba(0, 0, 0, .05);
            padding: .9rem 1.1rem; margin-bottom: .6rem;
            background: rgba(255, 255, 255, .8);
        }

        /* 指标卡片 */
        [data-testid="stMetric"] {
            background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
            padding: .7rem .9rem; box-shadow: 0 1px 4px rgba(0, 0, 0, .04);
        }

        /* 滚动条 */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #a5b4fc; }

        /* 侧边栏 */
        section[data-testid="stSidebar"] {
            background: #f8fafc; border-right: 1px solid #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


EMOTION_STYLES = {
    "positive": ("#dcfce7", "#15803d", "😊"),
    "negative": ("#fee2e2", "#b91c1c", "😢"),
    "neutral": ("#f1f5f9", "#475569", "😐"),
}

CATEGORY_STYLES = {
    "coding": ("#dbeafe", "#1e40af", "💻 编程"),
    "career": ("#fef3c7", "#92400e", "🎯 求职"),
    "emotional": ("#fce7f3", "#9d174d", "💙 情绪"),
    "chitchat": ("#f1f5f9", "#475569", "💬 闲聊"),
    "interview": ("#ede9fe", "#5b21b6", "🎤 面试"),
}


def render_badge(text, bg, fg):
    """渲染一个圆角彩色徽章（HTML span）。"""
    return (
        f'<span style="font-size:0.72rem;padding:2px 10px;border-radius:999px;'
        f'background:{bg};color:{fg};border:1px solid {fg}22;margin-right:6px;'
        f'display:inline-block;">{text}</span>'
    )


def render_message_badges(msg):
    """在消息气泡下方渲染 情绪/类别/时间 徽章行。"""
    badges = []
    if msg.get("emotion"):
        bg, fg, icon = EMOTION_STYLES.get(msg["emotion"], ("#f1f5f9", "#475569", ""))
        badges.append(render_badge(f"{icon} {msg['emotion']}", bg, fg))
    if msg.get("category"):
        bg, fg, label = CATEGORY_STYLES.get(
            msg["category"], ("#f1f5f9", "#475569", msg["category"])
        )
        badges.append(render_badge(label, bg, fg))
    if msg.get("time"):
        badges.append(render_badge(f"🕐 {msg['time']}", "#f8fafc", "#64748b"))
    if badges:
        st.markdown("".join(badges), unsafe_allow_html=True)


def render_message(msg):
    """渲染一条聊天气泡（含头像与徽章）。"""
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        render_message_badges(msg)


def stream_output(text, placeholder):
    """流式输出显示文本（带光标效果）。"""
    for i in range(0, len(text), 3):
        placeholder.markdown(text[: i + 3] + "▌")
        time.sleep(0.008)
    placeholder.markdown(text)
    return text


# 欢迎页示例问题：(图标, 标题, 示例消息)
WELCOME_PROMPTS = [
    ("💬", "随便聊聊", "今天心情一般，陪我聊聊天吧"),
    ("💻", "编程辅导", "用 Python 写一个二分查找，并帮我讲解一下思路"),
    ("📋", "求职建议", "我想找 AI 算法岗实习，帮我分析一下需要准备什么"),
    ("📊", "情绪复盘", "最近压力有点大，帮我分析下这段时间的情绪变化"),
]


def render_welcome():
    """无对话历史时的欢迎页：问候卡片 + 示例问题按钮。"""
    st.markdown(
        """
        <div class="welcome-card">
            <h3>👋 你好，我是 OfferPilot</h3>
            <p>你的多 Agent 求职领航员 —— 点击下方示例快速开始，或在底部输入框直接输入</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (icon, title, text) in enumerate(WELCOME_PROMPTS):
        with cols[i % 2]:
            if st.button(
                f"{icon}  {title}",
                key=f"welcome_{i}",
                width="stretch",
                help=text,
            ):
                st.session_state.pending_prompt = text
    st.caption("🎤 想练面试？切换顶部「模拟面试」视图，题目会按你的画像定制")


# ---------------- 主区域渲染 ----------------

def render_header():
    """渲染渐变头部横幅。"""
    st.markdown(
        """
        <div class="app-header">
            <h1>🧭 OfferPilot</h1>
            <p>你的多 Agent 求职领航员</p>
            <div>
                <span class="feature-badge">💻 编程辅导</span>
                <span class="feature-badge">🎯 求职建议</span>
                <span class="feature-badge">🎤 模拟面试</span>
                <span class="feature-badge">💙 情绪关怀</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats():
    """渲染实时统计卡片：本轮消息数 / 事实记忆 / 平均情绪 / 面试场次。"""
    profile = vector_store.get_user_profile(st.session_state.user_id)
    facts = vector_store.get_facts_with_metadata(st.session_state.user_id)
    trend = profile.get("emotional_trend", [])
    reports = profile.get("job_progress", {}).get("interview_reports", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💬 本轮消息", len(st.session_state.chat_history))
    c2.metric("📌 事实记忆", len(facts))
    if trend:
        avg = sum(trend) / len(trend)
        c3.metric("🎭 平均情绪", f"{avg:.2f}", delta=None if 0.4 <= avg <= 0.7 else f"{avg - 0.5:+.2f}")
    else:
        c3.metric("🎭 平均情绪", "—")
    c4.metric("🎤 面试场次", len(reports))


def render_chat_toolbar():
    """对话视图工具栏：生成日报 / 清空对话（面试模式已独立为顶部视图切换）。"""
    col_report, col_clear = st.columns(2)
    with col_report:
        if st.button("📊 生成日报", width="stretch"):
            with st.spinner("正在生成日报..."):
                try:
                    report = run_daily_report(st.session_state.user_id)
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": f"## 📊 今日日报\n\n{report}",
                            "emotion": None,
                            "time": datetime.now().strftime("%H:%M"),
                        }
                    )
                    save_conversation_direct()
                    st.rerun()
                except Exception as e:
                    st.error(f"日报生成失败: {str(e)}")
    with col_clear:
        if st.button("🗑️ 清空对话", width="stretch"):
            st.session_state.chat_history = []
            st.session_state.current_conversation_name = None
            st.rerun()


def process_pending_message(msg):
    """处理待生成的 assistant 占位消息：执行工作流并流式输出。"""
    # 待回复的用户消息 = 占位消息之前最近的一条 user 消息
    idx = st.session_state.chat_history.index(msg)
    prompt = next(
        (
            m["content"]
            for m in reversed(st.session_state.chat_history[:idx])
            if m["role"] == "user"
        ),
        "",
    )

    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        try:
            with st.spinner("🧠 思考中..."):
                result = run_workflow(
                    user_id=st.session_state.user_id,
                    message=prompt,
                    thread_id=st.session_state.thread_id,
                    interview_mode=st.session_state.interview_mode,
                    resume_summary=st.session_state.resume_brief or None,
                    jd_text=st.session_state.jd_text or None,
                )
            response = result.get("final_response", "抱歉，无法生成回复。")
            emotion = result.get("emotion_label", "neutral")
            category = result.get("message_category", "")

            if st.session_state.streaming_enabled:
                full_response = stream_output(response, placeholder)
            else:
                placeholder.markdown(response)
                full_response = response

            # 回写占位消息为完整消息
            msg.update(
                content=full_response,
                emotion=emotion,
                category=category,
                time=datetime.now().strftime("%H:%M"),
                pending=False,
            )
            render_message_badges(msg)
            save_conversation_direct()

        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            placeholder.error(error_msg)
            msg.update(content=error_msg, pending=False)
            logger.error(f"工作流执行失败: {e}")


def render_chat_tab():
    """渲染聊天 Tab：工具栏 + 消息历史（或欢迎页）。"""
    render_chat_toolbar()

    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg.get("pending"):
                process_pending_message(msg)
            else:
                render_message(msg)
        # 自动滚动到最新消息（Streamlit >= 1.40 支持）
        if hasattr(st, "autoscroll"):
            st.autoscroll()
    else:
        render_welcome()


def finalize_interview():
    """结束当前面试并记录：走「结束面试」→ 报告生成路径。

    - 无进行中的会话 → 仅关闭面试模式（零成本）
    - 会话进行中且有作答 → 经 run_workflow 触发 report 节点，
      报告自动写入画像 job_progress.interview_reports（跨场次记录），
      并追加到对话历史便于回看
    """
    st.session_state.interview_mode = False
    session = get_interview_session(
        st.session_state.user_id, st.session_state.thread_id
    )
    if not session.get("active"):
        return

    with st.spinner("正在生成本场面试报告..."):
        try:
            result = run_workflow(
                user_id=st.session_state.user_id,
                message="结束面试",
                thread_id=st.session_state.thread_id,
                interview_mode=True,
                resume_summary=st.session_state.resume_brief or None,
                jd_text=st.session_state.jd_text or None,
            )
            report = result.get("final_response", "")
            if report:
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": report,
                        "emotion": None,
                        "category": "interview",
                        "time": datetime.now().strftime("%H:%M"),
                    }
                )
                save_conversation_direct()
        except Exception as e:
            st.warning(f"面试收尾失败: {e}")
            logger.error(f"面试收尾失败: {e}")


def _on_view_change():
    """视图切换回调（rerun 前执行）：切到面试自动开启，切走自动收尾。"""
    new_view = st.session_state.view_selector
    prev_view = st.session_state.get("active_view", CHAT_VIEW)
    if new_view == prev_view:
        return

    if new_view == INTERVIEW_VIEW:
        st.session_state.interview_mode = True
    elif prev_view == INTERVIEW_VIEW:
        # 切走面试视图：关闭模式并标记收尾（LLM 调用放主流程，回调中不宜做重活）
        st.session_state.interview_mode = False
        st.session_state.interview_pending_finalize = True

    st.session_state.active_view = new_view


def render_interview_view():
    """渲染模拟面试视图：按画像定制的模式卡片 + 面试进度 + 问答历史。"""
    profile = vector_store.get_user_profile(st.session_state.user_id)
    session = get_interview_session(
        st.session_state.user_id, st.session_state.thread_id
    )

    # ---- 按用户画像定制的面试模式卡片 ----
    # 不同用户的面试方向由画像（目标岗位/学习目标）+ 首条消息推断，
    # 题目生成时结合画像定制，因此每个用户看到的问题与模式不同
    topic = (
        session.get("topic", "")
        if session.get("active")
        else _infer_interview_topic(profile, "")
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 面试方向", topic or "待定制")
    col2.metric("💼 目标岗位", profile.get("job_target", "未设置")[:10])
    col3.metric("📈 技能水平", profile.get("current_skill_level", "未设置")[:10])
    st.caption(
        "🤖 题目结合你的画像定制；回答后按得分动态追问薄弱点，"
        "问满自动生成报告并记录到画像（记忆管理页可回看历场表现）"
    )

    # ---- 会话进度 ----
    if session.get("active"):
        asked = session.get("asked_count", 0)
        total = session.get("max_questions", settings.INTERVIEW_MAX_QUESTIONS)
        st.progress(
            min(asked / total, 1.0) if total else 0.0,
            text=f"本场进度：已问 {asked}/{total} 题 · {session.get('topic', '')} 方向",
        )
        current_q = (session.get("questions") or [{}])[-1]
        if current_q.get("question") and session.get("awaiting_answer"):
            st.info(
                f"**当前题目**（{current_q.get('dimension', '综合能力')}）："
                f"{current_q['question'][:80]}..."
            )
        if st.button("🏁 提前结束并生成报告", width="stretch"):
            finalize_interview()
            st.rerun()
    elif session.get("questions"):
        st.success(
            "✅ 上一场面试已结束，报告已记录到你的画像。发送消息即可开始新一场！"
        )
    else:
        # ---- 定向面试配置区（未开始时显示） ----
        # 3A 长期复用：若画像里已存简历摘要，自动加载，用户不必每次重传
        if not st.session_state.resume_brief:
            profile_resume = profile.get("resume_summary") or {}
            if profile_resume.get("brief"):
                st.session_state.resume_brief = profile_resume["brief"]

        with st.expander("📋 定向面试配置（可选，上传简历+粘贴JD后题目更贴合）", expanded=True):
            up_col, paste_col = st.columns(2)
            with up_col:
                uploaded = st.file_uploader(
                    "上传简历（PDF/MD/TXT）", type=["pdf", "md", "txt"],
                    key="resume_upload",
                )
            with paste_col:
                pasted = st.text_area(
                    "或粘贴简历文本", height=90, key="resume_paste",
                )
            if st.button("🔍 解析简历", width="stretch"):
                try:
                    if uploaded is not None:
                        content = uploaded.read()
                        text = extract_text_from_file(content, uploaded.name)
                    elif pasted.strip():
                        text = pasted.strip()
                    else:
                        text = ""
                        st.warning("请上传文件或粘贴文本")
                    if text:
                        with st.spinner("正在解析简历..."):
                            summary = parse_resume(text)
                        st.session_state.resume_brief = summary.get("brief", "")
                        # 同步入画像长期复用
                        profile2 = vector_store.get_user_profile(st.session_state.user_id)
                        profile2["resume_summary"] = summary
                        vector_store.save_user_profile(st.session_state.user_id, profile2)
                        if summary.get("success"):
                            st.success(f"✅ 解析成功：{summary.get('summary','')}")
                        else:
                            st.warning("⚠️ LLM 解析失败，已用原文摘要兜底，仍可定向面试")
                except Exception as e:
                    st.error(f"解析失败：{e}")

            st.session_state.jd_text = st.text_area(
                "目标岗位 JD（粘贴全文，出题会优先考察 JD 技能要求）",
                value=st.session_state.jd_text,
                height=100,
                key="jd_input",
            )

            # 定向状态指示
            badges = []
            if st.session_state.resume_brief:
                badges.append("✅ 简历已就绪")
            if st.session_state.jd_text.strip():
                badges.append("✅ JD 已就绪")
            mode = "� 定向面试" if badges else "通识面试（按画像定制）"
            st.caption(" · ".join(badges) + f" → {mode}")

        st.info(
            "�� 欢迎来到模拟面试！" + (
                "已配置定向信息，发送消息开场将围绕简历项目与 JD 技能出题。" if badges
                else "发送任意消息（如「开始面试」）即可开场，方向按你的画像定制。"
            ) + " 切回「对话」视图会自动结束并生成报告。"
        )

    # ---- 面试问答历史（与对话共用一条时间线，含待生成消息处理） ----
    for msg in st.session_state.chat_history:
        if msg.get("pending"):
            process_pending_message(msg)
        else:
            render_message(msg)
    if hasattr(st, "autoscroll"):
        st.autoscroll()


def render_memory_panel():
    """渲染记忆管理面板：用户画像 / 事实记忆（可删除/纠正）/ 最近对话记忆。

    用户可控记忆：AI 提取的事实可能有错误或过时，
    用户在这里可以直接管理，消除"记忆黑盒"。
    """
    st.subheader("🧠 记忆管理")

    # ---- 用户画像与反思摘要 ----
    profile = vector_store.get_user_profile(st.session_state.user_id)

    with st.expander("👤 用户画像（含反思摘要）", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("学习目标", profile.get("learning_goal", "暂无")[:12])
        col2.metric("技能水平", profile.get("current_skill_level", "暂无")[:12])
        col3.metric("求职目标", profile.get("job_target", "暂无")[:12])
        if profile.get("reflection_summary"):
            st.info(f"**画像反思摘要**（{profile.get('reflection_timestamp', '')}）：\n\n"
                    f"{profile['reflection_summary']}")
        job_progress = profile.get("job_progress", {})
        reports = job_progress.get("interview_reports", [])
        if reports:
            latest = reports[-1]
            st.success(
                f"**最近一场模拟面试**：{latest.get('topic', '')} 方向，"
                f"平均分 {latest.get('avg_score', 0)}，"
                f"薄弱点：{'；'.join(latest.get('weaknesses', [])[:3]) or '无'}"
            )

    # ---- 事实记忆管理 ----
    st.markdown("#### 📌 事实记忆（AI 从对话中提取，可删除/纠正）")

    facts = vector_store.get_facts_with_metadata(st.session_state.user_id)
    if not facts:
        st.info("暂无事实记忆。多聊几轮后，AI 会自动提取你的学习目标、求职进展等事实。")
    else:
        st.caption(f"共 {len(facts)} 条。删除或纠正后立即生效，下次检索与画像反思将使用修正后的记忆。")
        for fact in facts:
            with st.expander(f"📌 {fact['text'][:50]}"):
                st.caption(f"提取时间: {fact['timestamp']} | ID: {fact['id'][:20]}...")
                col_edit, col_del = st.columns([3, 1])
                with col_edit:
                    new_text = st.text_input(
                        "纠正内容",
                        value=fact["text"],
                        key=f"edit_{fact['id']}",
                    )
                    if st.button("💾 保存纠正", key=f"save_{fact['id']}"):
                        if new_text.strip() and new_text.strip() != fact["text"]:
                            result = vector_store.update_fact(fact["id"], new_text.strip())
                            if result.get("success"):
                                st.success("已纠正！")
                                st.rerun()
                            else:
                                st.error(result.get("message", "纠正失败"))
                        else:
                            st.warning("内容未变化")
                with col_del:
                    if st.button("🗑️ 删除", key=f"del_{fact['id']}"):
                        result = vector_store.delete_fact(fact["id"])
                        if result.get("success"):
                            st.success("已删除！")
                            st.rerun()
                        else:
                            st.error(result.get("message", "删除失败"))

    # ---- 最近对话记忆（episode，只读） ----
    st.markdown("#### 💬 最近对话记忆（原文）")
    try:
        recent = vector_store.conversation_collection.get(
            where={"$and": [
                {"user_id": st.session_state.user_id},
                {"memory_type": "episode"},
            ]},
            limit=5,
        )
        if recent and recent["documents"]:
            for doc, meta in zip(
                reversed(recent["documents"]), reversed(recent["metadatas"])
            ):
                role_icon = "🧑" if meta.get("role") == "user" else "🤖"
                st.markdown(
                    f"{role_icon} `{meta.get('timestamp', '')}` "
                    f"({meta.get('category', '')})\n\n> {doc[:100]}"
                )
        else:
            st.info("暂无对话记忆")
    except Exception as e:
        st.error(f"读取对话记忆失败: {e}")


# ---------------- 侧边栏 ----------------

def render_sidebar():
    """渲染左侧边栏：用户画像、情绪仪表盘、记忆可视化、对话存储。"""
    with st.sidebar:
        st.title("⚙️ 设置面板")

        st.session_state.user_id = st.text_input(
            "用户 ID",
            value=st.session_state.user_id,
            help="输入你的用户标识，用于记忆和画像管理",
        )

        # thread_id 跟随 user_id（LangGraph 检查点/面试会话按线程隔离）
        if st.session_state.thread_id != f"thread_{st.session_state.user_id}":
            st.session_state.thread_id = f"thread_{st.session_state.user_id}"

        if st.button("➕ 新建对话", width="stretch"):
            create_new_conversation()

        # 当前对话名（首条消息发出前显示提示）
        current_name = st.session_state.current_conversation_name
        st.caption(
            f"📝 当前对话: {current_name}"
            if current_name
            else "📝 当前对话: 新对话（发送首条消息后自动命名）"
        )

        st.divider()
        st.subheader("👤 用户画像")

        profile = vector_store.get_user_profile(st.session_state.user_id)

        learning_goal = st.text_input(
            "学习目标",
            value=profile.get("learning_goal", "找算法实习"),
        )
        skill_level = st.text_input(
            "当前技能水平",
            value=profile.get("current_skill_level", "Python基础"),
        )
        job_target = st.text_input(
            "求职目标",
            value=profile.get("job_target", "AI开发"),
        )

        if st.button("💾 保存画像", width="stretch"):
            profile["learning_goal"] = learning_goal
            profile["current_skill_level"] = skill_level
            profile["job_target"] = job_target
            vector_store.save_user_profile(st.session_state.user_id, profile)
            st.success("画像已保存！")

        st.divider()
        st.subheader("🎭 情绪仪表盘")

        emotional_trend = profile.get("emotional_trend", [])
        if emotional_trend:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(emotional_trend) + 1)),
                    y=emotional_trend,
                    mode="lines+markers",
                    name="情绪得分",
                    line=dict(color="#6366f1", width=2.5, shape="spline"),
                    marker=dict(size=8, color="#8b5cf6",
                                line=dict(width=1, color="#fff")),
                    fill="tozeroy",
                    fillcolor="rgba(99, 102, 241, .12)",
                )
            )
            fig.update_layout(
                title="最近情绪趋势",
                xaxis_title="对话序号",
                yaxis_title="情绪得分",
                yaxis=dict(range=[0, 1]),
                height=250,
                margin=dict(l=20, r=20, t=40, b=30),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")

            avg_score = sum(emotional_trend) / len(emotional_trend)
            if avg_score < 0.4:
                st.warning(f"近期平均情绪: {avg_score:.2f} — 看起来你需要休息一下 💙")
            elif avg_score > 0.7:
                st.success(f"近期平均情绪: {avg_score:.2f} — 状态不错！🌟")
            else:
                st.info(f"近期平均情绪: {avg_score:.2f}")
        else:
            st.info("暂无情绪数据，开始对话后将自动记录。")

        st.divider()
        st.subheader("🧠 记忆可视化")

        if st.session_state.chat_history:
            last_user_msg = ""
            for msg in reversed(st.session_state.chat_history):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break

            if last_user_msg:
                memories = vector_store.retrieve_memories(
                    st.session_state.user_id, last_user_msg, top_k=3
                )
                if memories:
                    for i, mem in enumerate(memories, 1):
                        st.markdown(
                            f"**记忆 {i}** ({mem.get('timestamp', '')})\n\n"
                            f"情绪: {mem.get('emotion', '')} | "
                            f"类别: {mem.get('category', '')}\n\n"
                            f"> {mem.get('text', '')[:100]}..."
                        )
                else:
                    st.info("暂无相关记忆")
        else:
            st.info("开始对话后将显示相关记忆")

        st.divider()
        st.subheader("📚 已保存对话")

        if st.session_state.saved_conversations:
            for conv in st.session_state.saved_conversations:
                label = conv["name"]
                if len(label) > 38:
                    label = label[:38] + "…"
                with st.expander(f"💬 {label} · {conv['message_count']} 条"):
                    st.caption(f"保存时间: {conv['timestamp'][:19]}")
                    st.caption(f"用户 ID: {conv.get('user_id', '未知')}")
                    col_load, col_del = st.columns(2)
                    with col_load:
                        if st.button("📥 加载", key=f"load_{conv['filename']}",
                                     width="stretch"):
                            load_conversation(conv['filename'])
                    with col_del:
                        if st.button("🗑️ 删除", key=f"del_{conv['filename']}",
                                     width="stretch"):
                            delete_conversation(conv['filename'])
        else:
            st.info("暂无已保存的对话")


# ---------------- 输入与入口 ----------------

def handle_chat_input():
    """页面顶层聊天输入框：固定在页面最底部，始终位于所有消息下方。

    输入后仅追加 user 消息 + assistant 占位并触发 rerun，
    实际工作流执行由聊天 Tab 内的 process_pending_message 完成，
    保证新消息渲染在正确的位置。
    """
    prompt = st.session_state.pop("pending_prompt", None)
    typed = st.chat_input("输入消息，Enter 发送...")
    if typed:
        prompt = typed
    if not prompt:
        return

    now = datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append(
        {"role": "user", "content": prompt, "emotion": None, "time": now}
    )
    st.session_state.chat_history.append(
        {"role": "assistant", "content": "", "pending": True, "time": now}
    )
    st.rerun()


def main():
    """Streamlit 应用入口。"""
    st.set_page_config(
        page_title="OfferPilot",
        page_icon="🧭",
        layout="wide",
    )

    inject_custom_css()
    init_session_state()
    render_sidebar()
    render_header()
    render_stats()

    # 视图切换（替代原 Tabs）：切到「模拟面试」自动开启面试模式，
    # 切走自动结束并生成报告记录
    st.segmented_control(
        "视图切换",
        VIEW_OPTIONS,
        key="view_selector",
        default=CHAT_VIEW,
        on_change=_on_view_change,
        label_visibility="collapsed",
    )

    # 切走面试视图后的收尾（LLM 调用放主流程执行）
    if st.session_state.pop("interview_pending_finalize", False):
        finalize_interview()

    view = st.session_state.get("view_selector", CHAT_VIEW)
    if view == INTERVIEW_VIEW:
        render_interview_view()
    elif view == MEMORY_VIEW:
        render_memory_panel()
    else:
        render_chat_tab()

    # 顶层输入框：Streamlit 将其固定在页面最底部（所有消息下方）
    handle_chat_input()


if __name__ == "__main__":
    main()
