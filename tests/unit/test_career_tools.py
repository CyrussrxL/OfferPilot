"""求职工具 _core 本地层与 @tool 包装层（本地降级）单元测试。

测试环境 MCP_ENABLED=false（conftest 注入），@tool 层直接走本地实现，
不拉起 MCP Server 子进程。
"""

from companion_ai.tools.career_tools import (
    _evaluate_resume_core,
    _get_interview_questions_core,
    _get_job_requirements_core,
    get_job_requirements,
)


class TestGetJobRequirementsCore:
    def test_exact_match(self):
        result = _get_job_requirements_core("后端开发工程师")
        assert result["success"] is True
        assert result["matched"] is True
        assert result["position"] == "后端开发工程师"

    def test_fuzzy_direction_keyword(self):
        result = _get_job_requirements_core("后端")
        assert result["matched"] is True
        assert result["position"] == "后端开发工程师"

    def test_fuzzy_case_insensitive(self):
        result = _get_job_requirements_core("nlp相关岗位")
        assert result["matched"] is True
        assert result["position"] == "NLP 算法工程师"

    def test_unknown_position_lists_available(self):
        result = _get_job_requirements_core("宇航员")
        assert result["success"] is True
        assert result["matched"] is False
        assert len(result["available_positions"]) > 0
        assert "暂无" in result["message"]


class TestEvaluateResumeCore:
    def test_short_resume_penalized(self):
        result = _evaluate_resume_core("会Python")
        assert result["success"] is True
        assert 0 <= result["score"] < 60
        assert any("偏短" in imp for imp in result["improvements"])

    def test_rich_resume_scored(self):
        resume = (
            "熟练使用 Python、Java，掌握数据结构与算法，熟悉机器学习与深度学习，"
            "精通 SQL。项目经历：参与推荐系统开发与设计，实现召回优化，"
            "性能提升 30%，延迟减少 50ms。实习经历：后端开发实习，"
            "完成服务改进。GitHub: https://github.com/example 。" + "补充细节。" * 60
        )
        result = _evaluate_resume_core(resume)
        assert result["success"] is True
        assert result["score"] >= 60
        assert result["score_level"] in ("良好", "优秀")
        assert len(result["strengths"]) > 0

    def test_score_always_in_range(self):
        assert 0 <= _evaluate_resume_core("")["score"] <= 100
        assert 0 <= _evaluate_resume_core("x" * 10000)["score"] <= 100


class TestGetInterviewQuestionsCore:
    def test_normal_count(self):
        result = _get_interview_questions_core("AI", 3)
        assert result["success"] is True
        assert result["topic"] == "AI"
        assert len(result["questions"]) == 3

    def test_count_exceeds_bank_capped(self):
        result = _get_interview_questions_core("AI", 10)
        assert len(result["questions"]) <= 5  # 每个主题题库 5 题

    def test_unknown_topic_falls_back_ai(self):
        result = _get_interview_questions_core("不存在的话题", 2)
        assert result["topic"] == "AI"

    def test_questions_no_duplicate(self):
        result = _get_interview_questions_core("算法", 5)
        assert len(set(result["questions"])) == len(result["questions"])


class TestToolWrapperLocalFallback:
    """@tool 包装层：MCP_ENABLED=false 时走本地实现，可直接调用。"""

    def test_get_job_requirements_invoke(self):
        result = get_job_requirements.invoke({"position": "后端"})
        assert result["success"] is True
        assert result["matched"] is True
