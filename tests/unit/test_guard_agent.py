"""GuardAgent 分类逻辑单元测试。

分层分类（_classify_message）通过 mock vector_store 单例方法与
_llm_arbitrate 来验证各层行为，零 LLM/网络调用。
"""

import pytest

from companion_ai.agents.guard_agent import (
    _adjust_category_with_behavior,
    _classify_message,
    _classify_message_with_keywords,
    _detect_secondary_category,
    record_classification_correction,
)
from companion_ai.utils.config import settings


class TestKeywordClassification:
    def test_coding(self):
        category, confidence = _classify_message_with_keywords(
            "这段 python 代码有 bug，帮我调试一下"
        )
        assert category == "coding"
        assert confidence > 0

    def test_career(self):
        category, _ = _classify_message_with_keywords(
            "秋招投递简历后多久能收到面试通知"
        )
        assert category == "career"

    def test_emotional(self):
        category, _ = _classify_message_with_keywords("最近好焦虑，好累，想哭")
        assert category == "emotional"

    def test_chitchat_default(self):
        category, confidence = _classify_message_with_keywords("今天天气不错")
        assert (category, confidence) == ("chitchat", 0.5)

    def test_single_category_full_confidence(self):
        _, confidence = _classify_message_with_keywords("帮我调试 bug")
        assert confidence == 1.0  # 只命中 coding，占总分 100%


class TestClassifyMessage:
    """三层分类主方案：mock 向量分类，验证代码直出/仲裁/行为调整/兜底。"""

    def test_code_detected_directly(self):
        # 代码特征最高优先级，不进入向量分类
        category, confidence, arbitration = _classify_message(
            "```python\nprint('hi')\n```"
        )
        assert (category, confidence, arbitration) == ("coding", 0.95, False)

    def test_high_confidence_vector_direct(self, monkeypatch):
        from companion_ai.memory.vector_store import vector_store

        monkeypatch.setattr(
            vector_store,
            "classify_message_with_confidence",
            lambda text, top_k=3: ("career", 0.9),
        )
        category, confidence, arbitration = _classify_message("帮我看看简历")
        assert (category, confidence, arbitration) == ("career", 0.9, False)

    def test_low_confidence_arbitration(self, monkeypatch):
        from companion_ai.memory.vector_store import vector_store

        monkeypatch.setattr(
            vector_store,
            "classify_message_with_confidence",
            lambda text, top_k=3: ("chitchat", 0.4),
        )
        monkeypatch.setattr(
            "companion_ai.agents.guard_agent._llm_arbitrate",
            lambda text, cat, conf: ("emotional", 0.85),
        )
        category, confidence, arbitration = _classify_message("唉，太难了")
        assert (category, confidence, arbitration) == ("emotional", 0.85, True)

    def test_vector_failure_falls_back_to_keywords(self, monkeypatch):
        from companion_ai.memory.vector_store import vector_store

        def raise_error(text, top_k=3):
            raise RuntimeError("chroma down")

        monkeypatch.setattr(
            vector_store, "classify_message_with_confidence", raise_error
        )
        category, _, _ = _classify_message("这段代码报错了帮我调试")
        assert category == "coding"  # 关键词兜底

    def test_behavior_adjustment_applied(self, monkeypatch):
        from companion_ai.memory.vector_store import vector_store

        monkeypatch.setattr(
            vector_store,
            "classify_message_with_confidence",
            lambda text, top_k=3: ("career", 0.5),  # < 0.8 触发行为调整
        )
        behavior = {
            "emotional_tendency": "neutral",
            "is_late_night": True,
            "typing_speed": "fast",           # 规则 3：深夜 + 求职 + 快速 → emotional
            "message_length_category": "medium",
        }
        monkeypatch.setattr(settings, "GUARD_LLM_ARBITRATION", False)
        category, confidence, _ = _classify_message("offer 还没消息", behavior)
        assert category == "emotional"
        assert confidence == pytest.approx(0.65)  # 0.5 + 0.15


class TestAdjustCategoryWithBehavior:
    def test_late_night_career_fast_typing(self):
        behavior = {
            "emotional_tendency": "neutral", "is_late_night": True,
            "typing_speed": "fast", "message_length_category": "medium",
        }
        category, confidence = _adjust_category_with_behavior(
            "career", 0.5, "投递没回音", behavior
        )
        assert category == "emotional"
        assert confidence == pytest.approx(0.65)

    def test_thoughtful_keeps_category(self):
        behavior = {
            "emotional_tendency": "likely_thoughtful", "is_late_night": False,
            "typing_speed": "slow", "message_length_category": "long",
        }
        category, confidence = _adjust_category_with_behavior(
            "career", 0.9, "长消息", behavior
        )
        assert category == "career"
        assert confidence == pytest.approx(0.95)

    def test_emotional_tendency_with_indicators(self):
        behavior = {
            "emotional_tendency": "likely_emotional", "is_late_night": True,
            "typing_speed": "fast", "message_length_category": "short",
        }
        category, _ = _adjust_category_with_behavior(
            "chitchat", 0.5, "唉，烦死了", behavior
        )
        assert category == "emotional"

    def test_no_rule_matched_unchanged(self):
        behavior = {
            "emotional_tendency": "neutral", "is_late_night": False,
            "typing_speed": "normal", "message_length_category": "medium",
        }
        assert _adjust_category_with_behavior("career", 0.7, "你好", behavior) == (
            "career", 0.7
        )


class TestRecordClassificationCorrection:
    def test_empty_message_rejected(self):
        result = record_classification_correction("u1", "  ", "career")
        assert result["success"] is False
        assert result["seed_id"] == ""

    def test_correction_adds_seed(self):
        from companion_ai.memory.vector_store import vector_store

        before = vector_store.classification_collection.count()
        result = record_classification_correction(
            "u1", "测试错分消息：这道动态规划题怎么优化", "coding"
        )
        assert result["success"] is True
        assert result["seed_id"]
        after = vector_store.classification_collection.count()
        assert after == before + 1
        # 清理本次回流种子，避免污染种子库
        vector_store.classification_collection.delete(ids=[result["seed_id"]])


class TestDetectSecondaryCategory:
    """次意图检测：混合意图消息的主类别之外意图识别。"""

    # ---- 主类别任务型：情绪次意图 ----

    def test_career_primary_emotional_secondary(self):
        # 用户原始场景：开心 + 求职规划 → career 主，emotional 次
        assert _detect_secondary_category(
            "今天拿到 offer 了特别开心，帮我做下求职规划", "career"
        ) == "emotional"

    def test_coding_primary_emotional_secondary(self):
        assert _detect_secondary_category(
            "这道 dp 题调试了两小时终于过了，累但开心", "coding"
        ) == "emotional"

    # ---- 主类别任务型：coding/career 互检（需 >=2 命中）----

    def test_career_primary_coding_secondary(self):
        assert _detect_secondary_category(
            "投算法岗笔试考动态规划，我数据结构也不熟怎么办", "career"
        ) == "coding"

    def test_task_primary_single_other_hit_no_secondary(self):
        # 对方类别仅 1 个关键词命中 → 不误报
        assert _detect_secondary_category(
            "帮我看看这道链表题怎么写", "coding"
        ) == ""

    # ---- 主类别情绪/闲聊：任务次意图 ----

    def test_emotional_primary_career_strong_kw_single_hit(self):
        # 用户原始场景：向量分类判为 emotional 主类时，"求职"强词单次命中
        # 也必须识别出 career 次意图，避免求职需求被完全丢弃
        assert _detect_secondary_category(
            "我今天特别开心，同时需要你帮我做一下求职规划", "emotional"
        ) == "career"

    def test_emotional_primary_career_secondary(self):
        assert _detect_secondary_category(
            "最近好焦虑，秋招简历投了十几家都没回音", "emotional"
        ) == "career"

    def test_emotional_primary_coding_secondary(self):
        assert _detect_secondary_category(
            "心态崩了，这道递归排序的题看了半天还是报错", "emotional"
        ) == "coding"

    def test_chitchat_primary_task_secondary(self):
        assert _detect_secondary_category(
            "在吗，想问下 python 函数报错怎么调试", "chitchat"
        ) == "coding"

    # ---- 无次意图 ----

    def test_pure_task_no_secondary(self):
        assert _detect_secondary_category(
            "帮我讲讲二叉树的遍历方式", "coding"
        ) == ""

    def test_pure_emotional_no_secondary(self):
        assert _detect_secondary_category(
            "今天心情有点低落，想聊聊", "emotional"
        ) == ""

    def test_empty_text(self):
        assert _detect_secondary_category("", "chitchat") == ""
