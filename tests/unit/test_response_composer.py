"""ResponseComposer 单元测试：情绪关怀分级（纯函数）+ 节点拼接与记忆截断入库。"""

import pytest

from companion_ai.agents.response_composer import (
    _generate_emotion_care,
    _reset_care_rotation,
    general_chat,
    response_composer,
)
from companion_ai.memory.vector_store import vector_store

MODULE = "companion_ai.agents.response_composer"


@pytest.fixture(autouse=True)
def _reset_care_rotation_fixture():
    """每个测试前重置关怀模板轮换指针，保证断言确定性。"""
    _reset_care_rotation()
    yield


class TestGenerateEmotionCare:
    """情绪关怀分级：深度关怀 > 负面分级 > 正面分级 > 中性分类问候。"""

    def test_consecutive_negative_deep_care(self):
        # trend 按时间序，最近 3 次均为低分（<0.4）→ 深度关怀
        care = _generate_emotion_care("negative", 0.3, [0.5, 0.2, 0.3, 0.2], "chitchat")
        assert "💙" in care

    def test_trend_not_consecutive_no_deep_care(self):
        # 最近一次分数回升 → 不触发深度关怀
        care = _generate_emotion_care("negative", 0.3, [0.2, 0.3, 0.6], "chitchat")
        assert "💙" not in care

    def test_negative_high_score(self):
        assert "🤗" in _generate_emotion_care("negative", 0.8, [], "chitchat")

    def test_negative_mid_score(self):
        assert "😊" in _generate_emotion_care("negative", 0.5, [], "chitchat")

    def test_negative_low_score(self):
        assert "🫂" in _generate_emotion_care("negative", 0.3, [], "chitchat")

    def test_positive_high_score(self):
        assert "🌟" in _generate_emotion_care("positive", 0.8, [], "chitchat")

    def test_positive_mid_score(self):
        assert "👍" in _generate_emotion_care("positive", 0.5, [], "chitchat")

    def test_neutral_by_category(self):
        assert "📝" in _generate_emotion_care("neutral", 0.5, [], "coding")
        assert "💼" in _generate_emotion_care("neutral", 0.5, [], "career")
        assert "👋" in _generate_emotion_care("neutral", 0.5, [], "chitchat")


class TestCareRotation:
    """同层级模板轮换：连续触发换变体，绕圈回到第一条，层级 emoji 稳定。"""

    def test_rotation_changes_text(self):
        first = _generate_emotion_care("negative", 0.3, [], "chitchat")
        second = _generate_emotion_care("negative", 0.3, [], "chitchat")
        assert first != second  # 同层级连续触发 → 换变体

    def test_rotation_wraps_around(self):
        seen = [
            _generate_emotion_care("negative", 0.3, [], "chitchat")
            for _ in range(4)  # negative_low 只有 2 个变体，绕两圈
        ]
        assert seen[0] == seen[2]  # 绕一圈回到第一条
        assert seen[1] == seen[3]

    def test_rotation_keeps_tier_emoji(self):
        # 无论轮换到哪个变体，层级 emoji（分级语义锚点）不变
        for _ in range(4):
            assert "🫂" in _generate_emotion_care("negative", 0.3, [], "chitchat")
        for _ in range(4):
            assert "🌟" in _generate_emotion_care("positive", 0.8, [], "chitchat")


class TestSustainedLowMood:
    """持续低迷（移动均值）通道：正负交替的慢性低落也触发深度关怀。"""

    def test_moving_average_triggers_deep_care(self):
        # 末尾仅 1 次低分（不满足 3 连低），但窗口均值 0.4375 < 0.45 → 深度关怀
        care = _generate_emotion_care("negative", 0.35, [0.5, 0.4, 0.5, 0.35], "chitchat")
        assert "💙" in care

    def test_moving_average_above_threshold_not_triggered(self):
        # 窗口均值 0.5 ≥ 0.45 且无连续低落 → 不触发
        care = _generate_emotion_care("negative", 0.3, [0.6, 0.4, 0.6, 0.4], "chitchat")
        assert "💙" not in care

    def test_short_window_not_triggered_by_average(self):
        # 样本不足 4 条：均值再低也不走移动均值通道（且末尾仅 2 连低）
        care = _generate_emotion_care("neutral", 0.5, [0.6, 0.3, 0.3], "chitchat")
        assert "💙" not in care



class TestGeneralChat:
    def test_llm_response(self, mock_llm):
        mock_llm(MODULE, return_content="温暖的支持性回复")
        result = general_chat({
            "user_id": "u", "current_message": "我今天有点累",
            "emotion_label": "negative", "emotion_score": 0.3,
            "retrieved_memories": [], "user_profile": {},
        })
        assert result["general_response"] == "温暖的支持性回复"

    def test_llm_failure_fallback(self, mock_llm):
        llm = mock_llm(MODULE, return_content="")
        llm.invoke.side_effect = RuntimeError("LLM down")
        result = general_chat({"user_id": "u", "current_message": "你好"})
        assert "无法回复" in result["general_response"]


class TestResponseComposerNode:
    def _state(self, unique_user, **overrides):
        state = {
            "user_id": unique_user,
            "emotion_label": "neutral",
            "emotion_score": 0.5,
            "message_category": "chitchat",
            "user_profile": {},
            "general_response": "这是通用回复",
        }
        state.update(overrides)
        return state

    def _last_stored_assistant_text(self, unique_user):
        results = vector_store.conversation_collection.get(
            where={"$and": [{"user_id": unique_user}, {"role": "assistant"}]}
        )
        assert results["documents"], "assistant 回复未入库"
        # 取最新一条（documents 顺序不保证，用 metadata timestamp 排序）
        latest = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: x[1].get("timestamp", ""),
        )[-1]
        return latest[0]

    def test_neutral_no_care_prefix(self, unique_user):
        result = response_composer(self._state(unique_user))
        assert result["final_response"] == "这是通用回复"

    def test_negative_prepends_care(self, unique_user):
        result = response_composer(
            self._state(unique_user, emotion_label="negative", emotion_score=0.5)
        )
        assert result["final_response"].startswith(("😊", "🫂", "🤗", "💙"))
        assert "---" in result["final_response"]
        assert result["final_response"].endswith("这是通用回复")

    def test_interview_response_priority(self, unique_user):
        result = response_composer(
            self._state(
                unique_user,
                interview_mode=True,
                interview_response="第 1 题：请介绍你自己",
                message_category="career",
                career_response="求职建议",
            )
        )
        assert "第 1 题" in result["final_response"]
        assert "求职建议" not in result["final_response"]

    def test_category_routing_coding(self, unique_user):
        result = response_composer(
            self._state(
                unique_user,
                message_category="coding",
                coding_response="代码解析结果",
            )
        )
        assert result["final_response"] == "代码解析结果"

    def test_empty_response_fallback(self, unique_user):
        result = response_composer(self._state(unique_user, general_response=""))
        assert "无法处理" in result["final_response"]

    def test_long_response_truncated_in_storage(self, unique_user):
        """AI 回复入库截断至 500 字（+ 省略号），控制记忆存储与检索噪音。"""
        long_text = "很长的回复内容。" * 100  # ~800 字
        response_composer(self._state(unique_user, general_response=long_text))
        stored = self._last_stored_assistant_text(unique_user)
        assert len(stored) <= 503
        assert stored.endswith("...")

    def test_short_response_stored_intact(self, unique_user):
        response_composer(self._state(unique_user, general_response="简短回复"))
        stored = self._last_stored_assistant_text(unique_user)
        assert stored == "简短回复"


class TestSecondaryIntentResponse:
    """次意图回应段：混合意图消息的主回复后追加段。"""

    def _state(self, unique_user, **overrides):
        state = {
            "user_id": unique_user,
            "emotion_label": "neutral",
            "emotion_score": 0.5,
            "message_category": "chitchat",
            "user_profile": {},
            "general_response": "这是通用回复",
        }
        state.update(overrides)
        return state

    def test_emotional_secondary_neutral_appends_care(self, unique_user):
        # 主 career + 次 emotional + 中性情绪（前置关怀未触发）→ 追加轻量呼应
        result = response_composer(self._state(
            unique_user,
            message_category="career",
            secondary_category="emotional",
            career_response="这是求职规划建议",
        ))
        assert result["final_response"].startswith("这是求职规划建议")
        assert "💙" in result["final_response"]

    def test_emotional_secondary_positive_no_duplicate(self, unique_user):
        # 正面情绪已触发前置关怀 → 不再追加次意图段，避免重复
        result = response_composer(self._state(
            unique_user,
            emotion_label="positive",
            emotion_score=0.8,
            message_category="career",
            secondary_category="emotional",
            career_response="这是求职规划建议",
        ))
        assert result["final_response"].startswith("🌟")
        assert result["final_response"].count("💙") == 0

    def test_career_secondary_appends_hint(self, unique_user):
        # 主 emotional + 次 career → 追加求职引导段
        result = response_composer(self._state(
            unique_user,
            message_category="emotional",
            secondary_category="career",
            general_response="先抱抱你",
        ))
        assert result["final_response"].startswith("先抱抱你")
        assert "求职" in result["final_response"]
        assert "Gap" in result["final_response"]

    def test_coding_secondary_appends_hint(self, unique_user):
        result = response_composer(self._state(
            unique_user,
            message_category="emotional",
            secondary_category="coding",
            general_response="陪你聊聊",
        ))
        assert "💻" in result["final_response"]

    def test_no_secondary_unchanged(self, unique_user):
        # 无次意图 → 回复结构与原逻辑一致，无追加段
        result = response_composer(self._state(
            unique_user,
            message_category="career",
            career_response="纯求职建议",
        ))
        assert result["final_response"] == "纯求职建议"

    def test_secondary_same_as_primary_ignored(self, unique_user):
        # 次意图与主类别相同（异常防御）→ 不追加
        result = response_composer(self._state(
            unique_user,
            message_category="career",
            secondary_category="career",
            career_response="纯求职建议",
        ))
        assert result["final_response"] == "纯求职建议"
