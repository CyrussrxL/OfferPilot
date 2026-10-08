"""情感分析（关键词回退 + 模型路由）与行为分析器单元测试。

测试环境 SENTIMENT_FALLBACK_ENABLED=False（默认），sentiment_analyzer
单例直接走关键词方案，无模型加载、零网络调用；模型路由测试通过
monkeypatch 注入假 pipeline，同样零网络依赖。
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


def _patch_pipeline(monkeypatch, results=None, side_effect=None):
    """注入假 pipeline 并启用模型路径（monkeypatch 自动还原单例状态）。"""
    from companion_ai.emotion.sentiment_analyzer import sentiment_analyzer

    def fake_pipeline(text):
        if side_effect:
            raise side_effect
        return results

    monkeypatch.setattr(sentiment_analyzer, "use_fallback", False)
    monkeypatch.setattr(sentiment_analyzer, "pipeline", fake_pipeline)


class TestModelRouting:
    """模型可用时的统一路由：中英文均走模型，不再按语言分流。"""

    def test_chinese_uses_model_when_available(self, monkeypatch):
        _patch_pipeline(
            monkeypatch, results=[{"label": "negative", "score": 0.93}]
        )
        label, score = sentiment_analyzer_analyze("最近好焦虑，压力太大了")
        assert (label, score) == ("negative", 0.93)

    def test_english_uses_model(self, monkeypatch):
        _patch_pipeline(
            monkeypatch, results=[{"label": "positive", "score": 0.88}]
        )
        label, score = sentiment_analyzer_analyze("I got the offer, so happy!")
        assert (label, score) == ("positive", 0.88)

    def test_model_neutral_label_maps_to_half_score(self, monkeypatch):
        # 多语言模型的 neutral 是真实类别；分数统一为 0.5（与关键词方案一致）
        _patch_pipeline(
            monkeypatch, results=[{"label": "neutral", "score": 0.71}]
        )
        label, score = sentiment_analyzer_analyze("今天下午三点开会")
        assert (label, score) == ("neutral", 0.5)

    def test_model_label_case_insensitive(self, monkeypatch):
        _patch_pipeline(
            monkeypatch, results=[{"label": "POSITIVE", "score": 0.8}]
        )
        assert sentiment_analyzer_analyze("Great progress") == ("positive", 0.8)

    def test_model_low_confidence_maps_to_neutral(self, monkeypatch):
        # 三分类下 top-1 置信度 < 0.5（模型不确定）→ 中性兜底
        _patch_pipeline(
            monkeypatch, results=[{"label": "positive", "score": 0.4348}]
        )
        assert sentiment_analyzer_analyze("今天下午三点开会") == ("neutral", 0.5)

    def test_model_exception_falls_back_to_keywords(self, monkeypatch):
        _patch_pipeline(monkeypatch, side_effect=RuntimeError("model down"))
        label, score = sentiment_analyzer_analyze("最近好焦虑，压力太大了")
        assert label == "negative"
        assert score > 0.5


def _analyze_full(text):
    from companion_ai.emotion.sentiment_analyzer import sentiment_analyzer
    return sentiment_analyzer.analyze_full(text)


class TestValence:
    """效价（valence）计算：与置信度（score）语义分离。

    趋势的深度关怀通道（连续 3 次 < 0.4 / 5 次均值 < 0.45）按效价语义
    设计，负面样本的效价必须 < 0.4 才能触发。
    """

    def test_derive_valence_direct(self):
        from companion_ai.emotion.sentiment_analyzer import derive_valence

        assert derive_valence("positive", 0.8) == 0.8
        assert derive_valence("negative", 0.8) == 0.2
        assert derive_valence("neutral", 0.5) == 0.5

    def test_keyword_negative_valence_inverted(self):
        # 负面置信度越高 → 效价越低（1 - score），入趋势后可达 < 0.4
        label, score, valence = _analyze_full("最近好焦虑，压力太大了")
        assert label == "negative"
        assert valence == round(1 - score, 4)
        assert valence < 0.4

    def test_keyword_positive_valence_kept(self):
        label, score, valence = _analyze_full("今天面试过了，太开心了，感谢")
        assert label == "positive"
        assert valence == score

    def test_keyword_neutral_valence_half(self):
        assert _analyze_full("今天下午三点开会") == ("neutral", 0.5, 0.5)

    def test_empty_text_valence_half(self):
        assert _analyze_full("") == ("neutral", 0.5, 0.5)

    def test_model_full_distribution_valence(self, monkeypatch):
        # 完整分布：效价 = P(positive) + 0.5 × P(neutral) = 0.6 + 0.15
        from companion_ai.emotion.sentiment_analyzer import sentiment_analyzer

        def fake_pipeline(text, top_k=None):
            return [
                {"label": "positive", "score": 0.6},
                {"label": "neutral", "score": 0.3},
                {"label": "negative", "score": 0.1},
            ]

        monkeypatch.setattr(sentiment_analyzer, "use_fallback", False)
        monkeypatch.setattr(sentiment_analyzer, "pipeline", fake_pipeline)
        assert _analyze_full("今天状态不错") == ("positive", 0.6, 0.75)

    def test_model_single_result_valence_derived(self, monkeypatch):
        # pipeline 不支持 top_k 时退化为单结果 + derive_valence 近似
        _patch_pipeline(
            monkeypatch, results=[{"label": "negative", "score": 0.93}]
        )
        assert _analyze_full("最近好焦虑") == ("negative", 0.93, 0.07)


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
