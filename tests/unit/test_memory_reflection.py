"""记忆提取与画像反思单元测试（LLM 全 mock）。"""

import json

from companion_ai.memory.memory_reflection import extract_facts, reflect_on_profile
from companion_ai.memory.vector_store import vector_store
from companion_ai.utils.config import settings

MODULE = "companion_ai.memory.memory_reflection"


def _facts_content(facts):
    return json.dumps({"facts": facts}, ensure_ascii=False)


class TestExtractFacts:
    def test_short_message_skipped(self):
        assert extract_facts("u", "太短") == []

    def test_disabled_by_config(self, unique_user, monkeypatch):
        monkeypatch.setattr(settings, "MEMORY_FACT_EXTRACTION", False)
        assert extract_facts(unique_user, "这是一条足够长的测试消息") == []

    def test_extracts_and_stores(self, unique_user, mock_llm):
        mock_llm(
            MODULE, return_content=_facts_content(["用户目标是字节后端岗", "用户熟悉 LangGraph"])
        )
        new_facts = extract_facts(unique_user, "我在准备字节的后端面试，平时用 LangGraph 做项目")
        assert set(new_facts) == {"用户目标是字节后端岗", "用户熟悉 LangGraph"}
        stored = vector_store.get_all_facts(unique_user)
        assert set(stored) >= set(new_facts)

    def test_deduplicates_existing_facts(self, unique_user, mock_llm):
        from companion_ai.utils.helpers import get_timestamp

        vector_store.store_fact(unique_user, "用户熟悉 LangGraph", get_timestamp())
        mock_llm(MODULE, return_content=_facts_content(["用户熟悉 LangGraph", "用户目标是字节后端岗"]))
        new_facts = extract_facts(unique_user, "我在准备字节后端面试，用 LangGraph 做项目")
        # 已有事实被去重，只入库新事实
        assert new_facts == ["用户目标是字节后端岗"]
        assert vector_store.get_all_facts(unique_user).count("用户熟悉 LangGraph") == 1

    def test_llm_failure_returns_empty(self, unique_user, mock_llm):
        llm = mock_llm(MODULE, return_content="")
        llm.invoke.side_effect = RuntimeError("LLM down")
        assert extract_facts(unique_user, "这是一条足够长的消息内容") == []

    def test_non_list_facts_returns_empty(self, unique_user, mock_llm):
        mock_llm(MODULE, return_content=json.dumps({"facts": "不是列表"}))
        assert extract_facts(unique_user, "这是一条足够长的消息内容") == []

    def test_empty_facts_list(self, unique_user, mock_llm):
        mock_llm(MODULE, return_content=_facts_content([]))
        assert extract_facts(unique_user, "这道动态规划题怎么优化") == []


class TestReflectOnProfile:
    def test_insufficient_facts_skips(self, unique_user):
        result = reflect_on_profile(unique_user)
        assert result["success"] is False
        assert "不足" in result["message"]

    def test_reflection_updates_core_fields(self, unique_user, mock_llm):
        from companion_ai.utils.helpers import get_timestamp

        # 先准备：画像运行时字段 + 3 条事实
        vector_store.save_user_profile(unique_user, {
            "conversation_count": 10,
            "emotional_trend": [0.5, 0.6],
        })
        for fact in ["用户目标是字节后端岗", "用户熟悉 LangGraph", "用户在准备秋招"]:
            vector_store.store_fact(unique_user, fact, get_timestamp())

        mock_llm(MODULE, return_content=json.dumps({
            "learning_goal": "三个月内拿下字节后端 offer",
            "current_skill_level": "中级，熟悉 LangGraph",
            "job_target": "字节跳动后端开发",
            "reflection_summary": "用户正在秋招冲刺期。",
        }, ensure_ascii=False))

        result = reflect_on_profile(unique_user)
        assert result["success"] is True
        profile = result["profile"]
        assert profile["job_target"] == "字节跳动后端开发"
        assert profile["learning_goal"] == "三个月内拿下字节后端 offer"
        assert profile["reflection_summary"] == "用户正在秋招冲刺期。"
        # 运行时字段保留，不被反思覆盖
        assert profile["conversation_count"] == 10
        assert profile["emotional_trend"] == [0.5, 0.6]
        assert "reflection_timestamp" in profile

    def test_llm_failure_returns_failure(self, unique_user, mock_llm):
        from companion_ai.utils.helpers import get_timestamp

        for fact in ["事实一", "事实二", "事实三"]:
            vector_store.store_fact(unique_user, fact, get_timestamp())
        llm = mock_llm(MODULE, return_content="")
        llm.invoke.side_effect = RuntimeError("LLM down")
        result = reflect_on_profile(unique_user)
        assert result["success"] is False
        assert result["profile"] == {}
