"""LangGraph 工作流单元测试：条件路由、图单例、检查点。"""

import pytest

from companion_ai.graph import workflow
from companion_ai.utils.config import settings


class TestRouteByCategory:
    def test_interview_mode_priority(self):
        # 面试模式优先于消息类别（面试中任何输入都进状态机）
        assert workflow._route_by_category({
            "interview_mode": True, "message_category": "coding",
        }) == "interview"

    def test_interview_mode_disabled_falls_back(self, monkeypatch):
        monkeypatch.setattr(settings, "INTERVIEW_GRAPH_ENABLED", False)
        assert workflow._route_by_category({
            "interview_mode": True, "message_category": "coding",
        }) == "coding"

    def test_category_coding(self):
        assert workflow._route_by_category({"message_category": "coding"}) == "coding"

    def test_category_career(self):
        assert workflow._route_by_category({"message_category": "career"}) == "career"

    def test_category_emotional_to_general(self):
        assert workflow._route_by_category({
            "message_category": "emotional"
        }) == "general_chat"

    def test_category_chitchat_to_general(self):
        assert workflow._route_by_category({
            "message_category": "chitchat"
        }) == "general_chat"

    def test_missing_category_defaults_general(self):
        assert workflow._route_by_category({}) == "general_chat"


class TestRouteInterview:
    def test_report_decision(self):
        assert workflow._route_interview({
            "interview_session": {"decision": "report"}
        }) == "interview_report"

    def test_ask_decision(self):
        assert workflow._route_interview({
            "interview_session": {"decision": "ask"}
        }) == "interview_ask"

    def test_probe_and_next_loop_back(self):
        # probe（追问）与 next（下一题）都走回边继续出题
        assert workflow._route_interview({
            "interview_session": {"decision": "probe"}
        }) == "interview_ask"
        assert workflow._route_interview({
            "interview_session": {"decision": "next"}
        }) == "interview_ask"

    def test_no_session_defaults_ask(self):
        assert workflow._route_interview({}) == "interview_ask"


class TestCompileGraph:
    def test_process_level_singleton(self):
        g1 = workflow.compile_graph()
        g2 = workflow.compile_graph()
        assert g1 is g2

    def test_compiled_graph_contains_all_nodes(self):
        compiled = workflow.compile_graph()
        node_names = set(compiled.get_graph().nodes.keys())
        expected = {
            "guard", "memory", "interview_evaluate", "interview_ask",
            "interview_report", "coding", "career", "general_chat",
            "response_composer",
        }
        assert expected <= node_names

    def test_checkpointer_is_sqlite(self):
        """检查点优先 SqliteSaver 落盘（测试环境临时 db，conftest 注入）。"""
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpointer = workflow._create_checkpointer()
        assert isinstance(checkpointer, SqliteSaver)


class TestGetInterviewSession:
    """get_interview_session：从检查点读取面试会话状态。"""

    def test_unknown_thread_returns_empty(self):
        # 从未对话过的线程 → 无会话状态，返回空字典
        session = workflow.get_interview_session("ghost_user_xyz")
        assert session == {}


