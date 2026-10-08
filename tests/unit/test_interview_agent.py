"""InterviewAgent 面试状态机单元测试（LLM 全 mock）。"""

from companion_ai.agents.interview_agent import (
    _END_INTERVIEW_PATTERN,
    _infer_interview_topic,
    _new_session,
    _parse_json,
    interview_evaluate,
)
from companion_ai.utils.config import settings

EVAL_MODULE = "companion_ai.agents.interview_agent"


def _eval_content(score=80, weaknesses=None):
    import json

    return json.dumps({
        "score": score,
        "strengths": ["思路清晰"],
        "weaknesses": weaknesses if weaknesses is not None else ["深度不足"],
        "dimension_feedback": "总体不错",
    }, ensure_ascii=False)


class TestParseJson:
    def test_plain_json(self):
        assert _parse_json('{"score": 90}') == {"score": 90}

    def test_markdown_fenced(self):
        assert _parse_json('```json\n{"score": 90}\n```') == {"score": 90}

    def test_json_with_surrounding_text(self):
        assert _parse_json('好的，评估如下：{"score": 75} 以上') == {"score": 75}

    def test_invalid_returns_empty(self):
        assert _parse_json("完全不是 JSON") == {}

    def test_empty_returns_empty(self):
        assert _parse_json("") == {}


class TestEndInterviewPattern:
    def test_matches(self):
        for text in ["我想结束面试", "不面了", "到此为止吧", "请停止面试"]:
            assert _END_INTERVIEW_PATTERN.search(text), text

    def test_no_match(self):
        assert not _END_INTERVIEW_PATTERN.search("这道题我会，下一题")
        assert not _END_INTERVIEW_PATTERN.search("面试官您好")


class TestInferTopic:
    def test_from_profile_job_target(self):
        # "算法工程师" 命中 AI 主题关键词表（算法工程师≈AI 方向）
        assert _infer_interview_topic({"job_target": "算法工程师"}, "") == "AI"

    def test_from_message_algorithm(self):
        assert _infer_interview_topic({}, "我想刷算法和数据结构") == "算法"

    def test_from_message_python(self):
        assert _infer_interview_topic({}, "我想面 Python 后端") == "编程"

    def test_llm_direction(self):
        assert _infer_interview_topic({}, "大模型应用开发方向") == "AI"

    def test_default_ai(self):
        assert _infer_interview_topic({}, "开始吧") == "AI"


class TestNewSession:
    def test_defaults(self):
        session = _new_session({}, "开始面试")
        assert session["active"] is True
        assert session["asked_count"] == 0
        assert session["probe_count"] == 0
        assert session["awaiting_answer"] is False
        assert session["max_questions"] == settings.INTERVIEW_MAX_QUESTIONS
        assert session["questions"] == []
        assert session["topic"] == "AI"  # 无画像无方向 → 默认


class TestInterviewEvaluateDecisions:
    """决策矩阵：ask / report / probe / next / LLM 失败兜底。"""

    def _state(self, session=None, message="我的回答"):
        return {
            "user_id": "u_test",
            "current_message": message,
            "user_profile": {},
            "interview_session": session,
        }

    def test_no_session_creates_new(self):
        result = interview_evaluate(self._state(session=None))
        session = result["interview_session"]
        assert session["active"] is True
        assert session["decision"] == "ask"

    def test_inactive_session_creates_new(self):
        result = interview_evaluate(self._state(session={"active": False}))
        assert result["interview_session"]["active"] is True
        assert result["interview_session"]["decision"] == "ask"

    def test_not_awaiting_answer_asks(self):
        session = {"active": True, "awaiting_answer": False}
        result = interview_evaluate(self._state(session=session))
        assert result["interview_session"]["decision"] == "ask"

    def test_user_ends_interview(self):
        session = {
            "active": True, "awaiting_answer": True, "questions": [],
            "asked_count": 1, "probe_count": 0,
        }
        result = interview_evaluate(self._state(session=session, message="结束面试吧"))
        session = result["interview_session"]
        assert session["decision"] == "report"
        assert session["end_reason"] == "user_request"

    def test_low_score_triggers_probe(self, mock_llm):
        mock_llm(EVAL_MODULE, "_get_eval_llm", return_content=_eval_content(score=50))
        session = {
            "active": True, "awaiting_answer": True, "probe_count": 0,
            "asked_count": 1, "max_questions": 3,
            "questions": [{"question": "Q1", "dimension": "基础"}],
        }
        result = interview_evaluate(self._state(session=session))
        session = result["interview_session"]
        assert session["decision"] == "probe"
        assert session["probe_count"] == 1
        # 回填本轮回答与评估
        assert session["questions"][-1]["answer"] == "我的回答"
        assert session["questions"][-1]["evaluation"]["score"] == 50

    def test_all_questions_asked_reports(self, mock_llm):
        mock_llm(EVAL_MODULE, "_get_eval_llm", return_content=_eval_content(score=90))
        session = {
            "active": True, "awaiting_answer": True, "probe_count": 0,
            "asked_count": 3, "max_questions": 3,
            "questions": [{"question": "Q3", "dimension": "基础"}],
        }
        result = interview_evaluate(self._state(session=session))
        session = result["interview_session"]
        assert session["decision"] == "report"
        assert session["end_reason"] == "questions_completed"

    def test_good_score_moves_next(self, mock_llm):
        mock_llm(EVAL_MODULE, "_get_eval_llm", return_content=_eval_content(score=85))
        session = {
            "active": True, "awaiting_answer": True, "probe_count": 0,
            "asked_count": 1, "max_questions": 3,
            "questions": [{"question": "Q1", "dimension": "基础"}],
        }
        result = interview_evaluate(self._state(session=session))
        assert result["interview_session"]["decision"] == "next"

    def test_llm_failure_defaults_mid_score(self, mock_llm):
        llm = mock_llm(EVAL_MODULE, "_get_eval_llm", return_content="")
        llm.invoke.side_effect = RuntimeError("LLM down")
        session = {
            "active": True, "awaiting_answer": True, "probe_count": 0,
            "asked_count": 1, "max_questions": 3,
            "questions": [{"question": "Q1", "dimension": "基础"}],
        }
        result = interview_evaluate(self._state(session=session))
        session = result["interview_session"]
        # 兜底 50 分，但 weaknesses 为空 → 不触发追问，走 next
        assert session["decision"] == "next"
        assert session["questions"][-1]["evaluation"]["score"] == 50
        assert session["questions"][-1]["answer"] == "我的回答"

    def test_probe_already_used_no_second_probe(self, mock_llm):
        mock_llm(EVAL_MODULE, "_get_eval_llm", return_content=_eval_content(score=30))
        session = {
            "active": True, "awaiting_answer": True, "probe_count": 1,  # 已追问过
            "asked_count": 1, "max_questions": 3,
            "questions": [{"question": "Q1", "dimension": "基础"}],
        }
        result = interview_evaluate(self._state(session=session))
        assert result["interview_session"]["decision"] == "next"
