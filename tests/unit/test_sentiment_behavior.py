"""情感分析（关键词方案）与行为分析器单元测试。

测试环境 SENTIMENT_FALLBACK_ENABLED=False（默认），sentiment_analyzer
单例直接走关键词方案，无模型加载、零网络调用。
"""

from datetime import datetime

from companion_ai.agents.behavior_analyzer import UserBehaviorAnalyzer
from companion_ai.emotion.sentiment_analyzer import SentimentAnalyzer


class TestSentimentKeywords:
    def test_chinese_negative(self):
        label, score = sentiment_analyzer_analyze("最近好焦虑，压力太大了")
        assert label == "negative"
        assert score > 0.5

    def test_chinese_positive(self):
        label, score = sentiment_analyzer_analyze("今天面试过了，太开心了，感谢")
        assert label == "positive"
        assert score > 0.5

    def test_chinese_neutral(self):
        label, score = sentiment_analyzer_analyze("今天下午三点开会")
        assert (label, score) == ("neutral", 0.5)

    def test_empty_text(self):
        assert sentiment_analyzer_analyze("") == ("neutral", 0.5)
        assert sentiment_analyzer_analyze("   ") == ("neutral", 0.5)

    def test_english_negative_no_model(self):
        # 测试环境 pipeline 为 None → 英文也走关键词方案
        label, _ = sentiment_analyzer_analyze("I am so tired and stressed")
        assert label == "negative"

    def test_score_capped_at_095(self):
        # 大量负面词命中，分数上限 0.95
        text = "焦虑 烦躁 难过 崩溃 沮丧 失望 担心 害怕 无助 绝望 痛苦"
        _, score = sentiment_analyzer_analyze(text)
        assert score <= 0.95

    def test_mixed_equal_counts_neutral(self):
        # 正负命中数相同 → neutral（"好累"负面 vs "太好了"正面）
        label, score = sentiment_analyzer_analyze("好累，但是太好了")
        assert (label, score) == ("neutral", 0.5)


class TestContainsCode:
    def test_code_block(self):
        assert sentiment_contains_code("```python\nprint(1)\n```")

    def test_function_def(self):
        assert sentiment_contains_code("def foo():")

    def test_import_statement(self):
        assert sentiment_contains_code("import os")

    def test_plain_text(self):
        assert not sentiment_contains_code("你好，请问今天天气怎么样")


def sentiment_analyzer_analyze(text):
    from companion_ai.emotion.sentiment_analyzer import sentiment_analyzer
    return sentiment_analyzer.analyze(text)


def sentiment_contains_code(text):
    from companion_ai.emotion.sentiment_analyzer import sentiment_analyzer
    return sentiment_analyzer.contains_code(text)


class TestBehaviorLength:
    def test_short(self):
        assert UserBehaviorAnalyzer()._categorize_length(5) == "short"

    def test_medium(self):
        assert UserBehaviorAnalyzer()._categorize_length(20) == "medium"

    def test_long(self):
        assert UserBehaviorAnalyzer()._categorize_length(100) == "long"

    def test_boundaries(self):
        assert UserBehaviorAnalyzer()._categorize_length(10) == "short"
        assert UserBehaviorAnalyzer()._categorize_length(11) == "medium"
        assert UserBehaviorAnalyzer()._categorize_length(50) == "medium"
        assert UserBehaviorAnalyzer()._categorize_length(51) == "long"


class TestTimeOfDay:
    def test_morning(self):
        assert UserBehaviorAnalyzer()._get_time_of_day(_ts(10)) == "morning"

    def test_afternoon(self):
        assert UserBehaviorAnalyzer()._get_time_of_day(_ts(14)) == "afternoon"

    def test_evening(self):
        assert UserBehaviorAnalyzer()._get_time_of_day(_ts(20)) == "evening"

    def test_night_and_late_night(self):
        analyzer = UserBehaviorAnalyzer()
        assert analyzer._get_time_of_day(_ts(23)) == "night"
        assert analyzer._is_late_night(_ts(23))
        assert analyzer._is_late_night(_ts(3))
        assert not analyzer._is_late_night(_ts(10))


class TestTypingSpeed:
    def test_fast(self):
        assert UserBehaviorAnalyzer()._categorize_typing_speed(3) == "fast"

    def test_normal(self):
        assert UserBehaviorAnalyzer()._categorize_typing_speed(10) == "normal"

    def test_slow(self):
        assert UserBehaviorAnalyzer()._categorize_typing_speed(60) == "slow"


class TestEmotionalTendency:
    def test_fast_short_emotional(self):
        analyzer = UserBehaviorAnalyzer()
        behavior = {"typing_speed": "fast", "message_length_category": "short",
                    "is_late_night": False}
        assert analyzer._detect_emotional_tendency(behavior) == "likely_emotional"

    def test_late_night_short_emotional(self):
        analyzer = UserBehaviorAnalyzer()
        behavior = {"typing_speed": "unknown", "message_length_category": "short",
                    "is_late_night": True}
        assert analyzer._detect_emotional_tendency(behavior) == "likely_emotional"

    def test_slow_long_thoughtful(self):
        analyzer = UserBehaviorAnalyzer()
        behavior = {"typing_speed": "slow", "message_length_category": "long",
                    "is_late_night": False}
        assert analyzer._detect_emotional_tendency(behavior) == "likely_thoughtful"

    def test_neutral(self):
        analyzer = UserBehaviorAnalyzer()
        behavior = {"typing_speed": "normal", "message_length_category": "medium",
                    "is_late_night": False}
        assert analyzer._detect_emotional_tendency(behavior) == "neutral"


class TestAnalyzeBehaviorIntegration:
    def test_first_message_unknown_speed(self):
        analyzer = UserBehaviorAnalyzer()
        behavior = analyzer.analyze_behavior("u1", "你好", current_time=_ts(10))
        assert behavior["typing_speed"] == "unknown"
        assert behavior["time_since_last_message"] is None

    def test_second_message_interval(self):
        analyzer = UserBehaviorAnalyzer()
        t0 = _ts(10)
        analyzer.analyze_behavior("u2", "第一条消息", current_time=t0)
        behavior = analyzer.analyze_behavior("u2", "第二条", current_time=t0 + 3)
        assert behavior["time_since_last_message"] == 3
        assert behavior["typing_speed"] == "fast"

    def test_history_capped_at_50(self):
        analyzer = UserBehaviorAnalyzer()
        for i in range(60):
            analyzer.analyze_behavior("u3", "x", current_time=_ts(10) + i)
        assert len(analyzer.get_user_history("u3")) == 50

    def test_clear_history(self):
        analyzer = UserBehaviorAnalyzer()
        analyzer.analyze_behavior("u4", "hi", current_time=_ts(10))
        analyzer.clear_history("u4")
        assert analyzer.get_user_history("u4") == []


def _ts(hour: int) -> float:
    """构造本地时间当天指定小时的 Unix 时间戳（与实现使用本地时区一致）。"""
    return datetime(2026, 1, 1, hour, 0, 0).timestamp()
