"""utils/helpers.py 纯函数单元测试。"""

from companion_ai.utils.helpers import (
    emotion_intensity,
    format_memories,
    format_user_profile,
    get_timestamp,
    safe_json_dumps,
    safe_json_loads,
    truncate_text,
)


class TestTruncateText:
    def test_short_text_unchanged(self):
        assert truncate_text("短文本", 500) == "短文本"

    def test_exact_boundary_unchanged(self):
        text = "a" * 500
        assert truncate_text(text, 500) == text

    def test_long_text_truncated_with_ellipsis(self):
        result = truncate_text("b" * 600, 500)
        assert len(result) == 503
        assert result.endswith("...")
        assert result[:500] == "b" * 500

    def test_default_max_length(self):
        assert len(truncate_text("c" * 600)) == 503


class TestSafeJson:
    def test_loads_valid(self):
        assert safe_json_loads('{"a": 1}') == {"a": 1}

    def test_loads_invalid_returns_empty(self):
        assert safe_json_loads("not json") == {}

    def test_loads_none_returns_empty(self):
        assert safe_json_loads(None) == {}

    def test_dumps_dict(self):
        assert safe_json_loads(safe_json_dumps({"key": "值"})) == {"key": "值"}

    def test_dumps_unserializable_returns_empty(self):
        assert safe_json_dumps(lambda x: x) == "{}"


class TestEmotionIntensity:
    def test_low(self):
        assert emotion_intensity(0.3) == "低"

    def test_medium(self):
        assert emotion_intensity(0.5) == "中"

    def test_high(self):
        assert emotion_intensity(0.8) == "高"

    def test_boundaries(self):
        assert emotion_intensity(0.4) == "中"  # 0.4 不属于低
        assert emotion_intensity(0.7) == "高"  # 0.7 不属于中


class TestFormatMemories:
    def test_empty(self):
        assert format_memories([]) == "暂无相关历史记忆。"

    def test_fact_marked(self):
        result = format_memories([{"memory_type": "fact", "text": "用户熟悉 Python"}])
        assert "[用户事实1] 用户熟悉 Python" in result

    def test_episode_marked_with_timestamp(self):
        result = format_memories([{
            "memory_type": "episode", "text": "在学 LangGraph",
            "timestamp": "2026-01-01 10:00:00", "emotion": "neutral",
        }])
        assert "[历史对话1] (2026-01-01 10:00:00, 情绪:neutral) 在学 LangGraph" in result

    def test_mixed_numbering(self):
        result = format_memories([
            {"memory_type": "fact", "text": "事实A"},
            {"memory_type": "episode", "text": "对话B", "timestamp": "", "emotion": "neutral"},
            {"memory_type": "fact", "text": "事实C"},
        ])
        assert "[用户事实1]" in result and "[用户事实2]" in result
        assert "[历史对话1]" in result


class TestFormatUserProfile:
    def test_empty(self):
        assert format_user_profile({}) == "暂无用户画像信息。"

    def test_label_mapping(self):
        result = format_user_profile({"learning_goal": "成为算法工程师"})
        assert "学习目标: 成为算法工程师" in result

    def test_emotional_trend_list_formatted(self):
        result = format_user_profile({"emotional_trend": [0.5, 0.6]})
        assert "0.50, 0.60" in result

    def test_unknown_key_kept_as_is(self):
        result = format_user_profile({"custom_key": "v"})
        assert "custom_key: v" in result


def test_get_timestamp_format():
    ts = get_timestamp()
    assert len(ts) == 19  # "YYYY-MM-DD HH:MM:SS"
    assert ts[4] == "-" and ts[10] == " " and ts[13] == ":"
