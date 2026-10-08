"""
LangGraph 工作流定义 —— workflow.py

职责：
  定义多 Agent 协作的完整状态图，包括节点、边和条件路由。

工作流设计：
  1. guard → 情感分析 + 消息分类
  2. memory → 检索记忆 + 更新记忆
  3. 根据 message_category 条件路由：
     - interview_mode → interview_evaluate（循环面试状态机入口）
     - coding → coding 节点
     - career → career 节点
     - emotional/chitchat → general_chat 节点
  4. 面试状态机（带条件回边的循环对话图）：
     - interview_evaluate →(ask/probe)→ interview_ask → response_composer
     - interview_evaluate →(report)→ interview_report → response_composer
  5. 最后进入 response_composer 节点：情绪关怀 + 拼接最终回复

设计理由：
  - 使用 LangGraph 的 StateGraph 构建有向图，每个 Agent 是一个节点。
  - 条件边（add_conditional_edges）实现基于消息类别的动态路由，
    这是多 Agent 协作的核心——不同类型的消息由不同的专家 Agent 处理。
  - 面试状态机是图中的循环对话子流程：evaluate → ask 的条件回边 +
    检查点持久化的 interview_session，让多轮面试在跨轮次间连续，
    支持动态追问（probe）与报告生成（report）两种转移。
  - response_composer 节点在所有路径之后执行，整合情绪关怀与专业回复，
    确保每条消息都能获得有温度的响应。
  - 编译后的图与 MemorySaver 检查点均为进程级单例：面试会话状态
    （interview_session）依赖检查点跨轮持久化，重复编译会丢失会话。
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from companion_ai.agents.career_agent import career_agent
from companion_ai.agents.coding_agent import coding_agent
from companion_ai.agents.guard_agent import guard_agent
from companion_ai.agents.interview_agent import (
    interview_ask,
    interview_evaluate,
    interview_report,
)
from companion_ai.agents.memory_agent import memory_agent
from companion_ai.agents.response_composer import general_chat, generate_daily_report, response_composer
from companion_ai.graph.state import State
from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger

# 进程级单例：编译图与检查点存储
_compiled_graph = None


def _route_by_category(state: State) -> Literal["interview", "coding", "career", "general_chat"]:
    """
    条件路由函数：根据 interview_mode / message_category 决定下一个节点。

    路由规则：
      - interview_mode 且面试图启用 → interview（循环面试状态机）
      - coding → coding 节点
      - career → career 节点
      - emotional / chitchat → general_chat 节点

    设计理由：
      条件边是 LangGraph 多 Agent 架构的关键，
      它让不同类型的消息被路由到最合适的专家 Agent，
      避免了单 Agent 需要处理所有类型消息的复杂性。
      面试模式优先于消息类别：面试中用户的任何输入
      （回答/追问回应/结束指令）都应进入面试状态机。
    """
    if state.get("interview_mode") and settings.INTERVIEW_GRAPH_ENABLED:
        logger.info("路由决策 | -> interview（循环面试状态机）")
        return "interview"

    category = state.get("message_category", "chitchat")
    logger.info(f"路由决策 | category={category}")

    if category == "coding":
        return "coding"
    elif category == "career":
        return "career"
    else:
        return "general_chat"


def _route_interview(state: State) -> Literal["interview_ask", "interview_report"]:
    """
    面试状态机条件边：根据 evaluate 节点的决策转移。

      - ask / next / probe → interview_ask（出题或动态追问，回边）
      - report → interview_report（生成报告，面试结束）
    """
    decision = (state.get("interview_session") or {}).get("decision", "ask")
    if decision == "report":
        return "interview_report"
    return "interview_ask"


def build_graph() -> StateGraph:
    """
    构建多 Agent 协作的 LangGraph 状态图。

    节点：
      - guard: 入口 Agent，情感分析 + 分类
      - memory: 记忆 Agent，检索 + 存储
      - interview_evaluate: 面试状态机决策节点（评估回答/判定转移）
      - interview_ask: 面试出题/动态追问节点
      - interview_report: 面试报告生成节点
      - coding: 编程辅导 Agent
      - career: 求职成长 Agent
      - general_chat: 通用对话节点
      - response_composer: 响应合成器，情绪关怀 + 拼接最终回复

    边：
      - guard → memory（固定边）
      - memory → interview/coding/career/general_chat（条件边）
      - interview_evaluate → interview_ask / interview_report（条件回边）
      - 各业务节点 → response_composer（固定边）
      - response_composer → END（固定边）
    """
    graph = StateGraph(State)

    graph.add_node("guard", guard_agent)
    graph.add_node("memory", memory_agent)
    graph.add_node("interview_evaluate", interview_evaluate)
    graph.add_node("interview_ask", interview_ask)
    graph.add_node("interview_report", interview_report)
    graph.add_node("coding", coding_agent)
    graph.add_node("career", career_agent)
    graph.add_node("general_chat", general_chat)
    graph.add_node("response_composer", response_composer)

    graph.set_entry_point("guard")

    graph.add_edge("guard", "memory")

    graph.add_conditional_edges(
        "memory",
        _route_by_category,
        {
            "interview": "interview_evaluate",
            "coding": "coding",
            "career": "career",
            "general_chat": "general_chat",
        },
    )

    # 面试状态机的条件回边：评估后转移到追问/下一题，或生成报告
    graph.add_conditional_edges(
        "interview_evaluate",
        _route_interview,
        {
            "interview_ask": "interview_ask",
            "interview_report": "interview_report",
        },
    )

    graph.add_edge("interview_ask", "response_composer")
    graph.add_edge("interview_report", "response_composer")
    graph.add_edge("coding", "response_composer")
    graph.add_edge("career", "response_composer")
    graph.add_edge("general_chat", "response_composer")

    graph.add_edge("response_composer", END)

    logger.info("LangGraph 工作流构建完成")
    return graph


def compile_graph():
    """
    编译工作流图（进程级单例），附加 SqliteSaver 检查点。

    单例的必要性：面试会话状态（interview_session）依赖检查点跨轮
    持久化——同一 thread_id 的多次 invoke 构成一场连续的模拟面试，
    每次重新编译（新检查点）会丢失会话状态。

    检查点选型：
      - SqliteSaver：检查点落盘（CHECKPOINT_DB_PATH），进程重启后
        同一 thread_id 的面试会话可完整恢复（MemorySaver 是纯内存，
        重启即丢）。
      - 依赖未安装时降级 MemorySaver，保证最小可运行。
    """
    global _compiled_graph

    if _compiled_graph is None:
        checkpointer = _create_checkpointer()
        graph = build_graph()
        _compiled_graph = graph.compile(checkpointer=checkpointer)
        logger.info("LangGraph 工作流编译完成（进程级单例，含检查点）")
    return _compiled_graph


def _create_checkpointer():
    """创建检查点存储：优先 SqliteSaver（持久化），失败降级 MemorySaver。"""
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(
            settings.CHECKPOINT_DB_PATH, check_same_thread=False
        )
        logger.info(f"检查点存储: SqliteSaver ({settings.CHECKPOINT_DB_PATH})")
        return SqliteSaver(conn)
    except Exception as e:
        from langgraph.checkpoint.memory import MemorySaver

        logger.warning(f"SqliteSaver 不可用，降级 MemorySaver（重启丢会话）: {e}")
        return MemorySaver()


def run_workflow(
    user_id: str,
    message: str,
    thread_id: str = None,
    interview_mode: bool = False,
) -> dict:
    """
    运行完整的工作流，返回最终状态。

    Args:
        user_id: 用户唯一标识
        message: 用户消息
        thread_id: 对话线程 ID（用于检查点恢复，面试会话依赖它跨轮连续）
        interview_mode: 是否开启模拟面试模式

    Returns:
        包含 final_response 的状态字典
    """
    compiled = compile_graph()

    if thread_id is None:
        thread_id = f"thread_{user_id}"

    initial_state = {
        "user_id": user_id,
        "current_message": message,
        "interview_mode": interview_mode,
    }
    # 面试模式关闭时重置会话：避免 checkpoint 中旧 session 残留，
    # 导致用户重开面试时把关闭期间的闲聊误当作上一题的回答评估
    if not interview_mode:
        initial_state["interview_session"] = {"active": False, "awaiting_answer": False}

    config = {"configurable": {"thread_id": thread_id}}

    logger.info(f"工作流启动 | user={user_id} | thread={thread_id}")

    result = compiled.invoke(initial_state, config=config)

    logger.info(
        f"工作流完成 | user={user_id} | "
        f"category={result.get('message_category', '')} | "
        f"emotion={result.get('emotion_label', '')}"
    )

    return result


def get_interview_session(user_id: str, thread_id: str = None) -> dict:
    """
    读取指定用户当前线程的面试会话状态（从检查点）。

    供前端展示面试进度（已问题数/当前题目/是否进行中），
    以及切换视图时判断是否需要收尾（生成报告）。

    Args:
        user_id: 用户唯一标识
        thread_id: 对话线程 ID，默认 thread_{user_id}

    Returns:
        interview_session 字典；无会话或读取失败时返回空字典
    """
    compiled = compile_graph()
    if thread_id is None:
        thread_id = f"thread_{user_id}"
    try:
        snapshot = compiled.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        return (snapshot.values or {}).get("interview_session") or {}
    except Exception as e:
        logger.warning(f"读取面试会话状态失败: {e}")
        return {}


def run_daily_report(user_id: str) -> str:
    """
    生成用户日报。

    Args:
        user_id: 用户唯一标识

    Returns:
        日报文本
    """
    state = {"user_id": user_id}
    result = generate_daily_report(state)
    return result.get("daily_report", "日报生成失败")
