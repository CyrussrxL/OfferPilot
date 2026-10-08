"""简历解析 + 定向面试注入单元测试（LLM 全 mock，零网络）。"""

import json

from companion_ai.agents.interview_agent import (
    _infer_interview_topic,
    _new_session,
    interview_evaluate,
)
from companion_ai.tools.resume_parser import build_resume_brief, parse_resume

RESUME_MODULE = "companion_ai.tools.resume_parser"


# ---------------- 简历解析 ----------------


class TestBuildResumeBrief:
    def test_success_summary(self):
        summary = {
            "success": True,
            "summary": "3年Python后端",
            "target_role": "后端开发",
            "skills": ["Python", "FastAPI"],
            "projects": ["推荐系统: 召回排序"],
            "highlights": ["QPS提升3倍"],
        }
        brief = build_resume_brief(summary)
        assert "3年Python后端" in brief
        assert "后端开发" in brief
        assert "Python" in brief
        assert "推荐系统" in brief
        assert "QPS提升3倍" in brief

    def test_empty_returns_empty(self):
        assert build_resume_brief({}) == ""

    def test_failure_returns_raw_excerpt(self):
        summary = {"success": False, "raw_excerpt": "原始简历文本片段"}
        assert build_resume_brief(summary) == "原始简历文本片段"

    def test_truncated_to_500(self):
        summary = {
            "success": True,
            "summary": "X" * 600,
            "skills": [],
            "projects": [],
            "highlights": [],
        }
        assert len(build_resume_brief(summary)) <= 500


class TestParseResume:
    def test_success(self, mock_llm):
        content = json.dumps({
            "summary": "应届算法",
            "target_role": "算法工程师",
            "skills": ["Python", "PyTorch"],
            "projects": ["图像分类项目"],
            "highlights": ["竞赛获奖"],
        }, ensure_ascii=False)
        mock_llm(RESUME_MODULE, "_get_llm", return_content=content)
        result = parse_resume("某人的简历文本内容")
        assert result["success"] is True
        assert result["target_role"] == "算法工程师"
        assert "Python" in result["skills"]
        assert result["brief"]  # 预构建摘要非空
        assert "算法" in result["brief"]

    def test_empty_text_returns_failure(self):
        result = parse_resume("")
        assert result["success"] is False
        assert result["brief"] == ""

    def test_llm_failure_fallback_to_excerpt(self, mock_llm):
        llm = mock_llm(RESUME_MODULE, "_get_llm", return_content="not json at all")
        llm.invoke.side_effect = Exception("网络故障")
        result = parse_resume("一段足够长的简历原文内容用于兜底")
        assert result["success"] is False
        assert result["raw_excerpt"]  # 兜底原文截断
        assert result["brief"] == result["raw_excerpt"]


# ---------------- 定向面试主题推断 ----------------


class TestInferTopicWithJD:
    def test_jd_drives_topic(self):
        profile = {"job_target": "", "learning_goal": ""}
        jd = "岗位：后端开发工程师，要求熟悉 Java、分布式架构"
        topic = _infer_interview_topic(profile, "", jd_text=jd)
        assert topic == "编程"  # "后端"/"开发" 命中编程主题

    def test_jd_system_design(self):
        profile = {"job_target": "", "learning_goal": ""}
        jd = "高并发系统设计，分布式架构"
        assert _infer_interview_topic(profile, "", jd_text=jd) == "系统设计"

    def test_no_jd_falls_back_to_profile(self):
        profile = {"job_target": "算法工程师", "learning_goal": ""}
        assert _infer_interview_topic(profile, "") == "AI"


# ---------------- _new_session 定向上下文注入 ----------------


class TestNewSessionDirected:
    def test_stores_resume_and_jd(self):
        session = _new_session(
            {"job_target": "算法工程师", "learning_goal": ""},
            "开始面试",
            resume_summary="画像：3年Python后端",
            jd_text="岗位：后端开发",
        )
        assert session["resume_summary"] == "画像：3年Python后端"
        assert session["jd_text"] == "岗位：后端开发"
        assert session["active"] is True

    def test_backward_compat_no_context(self):
        session = _new_session({"job_target": ""}, "开始面试")
        assert session["resume_summary"] == ""
        assert session["jd_text"] == ""
        # 旧调用方不传定向参数也能正常工作


# ---------------- interview_evaluate 开启定向会话 ----------------


class TestEvaluateOpensDirectedSession:
    def _state(self, resume="", jd=""):
        return {
            "user_id": "test_user",
            "current_message": "开始面试",
            "user_profile": {"job_target": "算法工程师", "learning_goal": ""},
            "interview_session": {},
            "resume_summary": resume,
            "jd_text": jd,
        }

    def test_directed_session_injects_context(self):
        state = self._state(resume="画像：3年Python", jd="岗位：算法工程师")
        result = interview_evaluate(state)
        session = result["interview_session"]
        assert session["active"] is True
        assert session["resume_summary"] == "画像：3年Python"
        assert session["jd_text"] == "岗位：算法工程师"
        assert session["decision"] == "ask"

    def test_no_context_backward_compat(self):
        result = interview_evaluate(self._state())
        session = result["interview_session"]
        assert session["active"] is True
        assert session["resume_summary"] == ""
        assert session["jd_text"] == ""
