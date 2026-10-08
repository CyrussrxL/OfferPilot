"""
情感分析模块 —— SentimentAnalyzer

职责：
  对用户输入文本进行情感分析，返回情感标签（positive/negative/neutral）和情感分数（0~1）。

设计理由：
  - 情感模型由用户自选（SENTIMENT_MODEL_NAME）：任意 HuggingFace 文本
    分类模型均可（需输出 positive/negative(/neutral) 标签）。多语言
    三分类模型（如 lxyuan/distilbert-base-multilingual-cased-sentiments-student）
    可中英文统一处理，无需按语言分流。
  - 提供基于关键词的本地回退方案：未配置模型、模型加载失败（网络问题、
    依赖缺失）或 SENTIMENT_FALLBACK_ENABLED=False（快速启动）时兜底，
    基于中英文情感关键词匹配，覆盖常见的情绪表达。
"""

import os
import re
from typing import Tuple

from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


def derive_valence(label: str, score: float) -> float:
    """
    由 (label, score) 推导情绪效价（0=负面, 0.5=中性, 1=正面）。

    score 语义是标签置信度/强度（negative 0.9 = 强负面），效价需要
    反转负面方向：效价 = 1 - score。关键词回退路径、模型单结果退化
    路径以及旧状态兼容（无 emotion_valence 的历史检查点）时使用。
    """
    if label == "positive":
        return round(score, 4)
    if label == "negative":
        return round(1 - score, 4)
    return 0.5


class SentimentAnalyzer:
    """
    情感分析器，支持 transformers 预训练模型和本地关键词回退两种模式。
    """

    NEGATIVE_KEYWORDS_CN = [
        "焦虑", "累", "烦", "难过", "崩溃", "压力", "抑郁", "沮丧",
        "失望", "担心", "害怕", "疲惫", "无助", "绝望", "痛苦",
        "心累", "烦躁", "郁闷", "低落", "丧", "不想", "放弃",
        "好累", "好烦", "好难", "太难", "好大", "受不了", "烦死",
    ]
    POSITIVE_KEYWORDS_CN = [
        "开心", "谢谢", "高兴", "棒", "喜欢", "兴奋", "满足",
        "成功", "进步", "感谢", "不错", "厉害", "优秀", "自信",
        "愉快", "舒适", "期待", "有趣", "收获", "很好", "太好了",
        "好开心", "好棒",
    ]
    NEGATIVE_KEYWORDS_EN = [
        "anxious", "tired", "annoyed", "sad", "depressed", "stressed",
        "frustrated", "worried", "exhausted", "helpless", "hopeless",
        "painful", "overwhelmed", "upset", "angry", "afraid", "bad",
        "terrible", "awful", "miserable",
    ]
    POSITIVE_KEYWORDS_EN = [
        "happy", "thanks", "great", "good", "love", "excited", "satisfied",
        "success", "progress", "grateful", "nice", "awesome", "excellent",
        "confident", "joyful", "comfortable", "looking forward", "fun",
        "amazing", "wonderful",
    ]

    def __init__(self):
        self.pipeline = None
        self.use_fallback = False
        self._load_model()

    def _load_model(self):
        """
        尝试加载 transformers 预训练情感分析 pipeline。
        若加载失败（网络问题、依赖缺失等），自动切换到关键词回退模式。

        模型由用户自选（SENTIMENT_MODEL_NAME，任意 HuggingFace 文本分类
        模型，需输出 positive/negative(/neutral) 标签），仓库不预置确定模型。

        SENTIMENT_FALLBACK_ENABLED 的语义：
          - True（默认）：先尝试加载模型，失败后回退到关键词方案
          - False：直接使用关键词方案，跳过模型加载（适用于无网络或快速启动场景）
        """
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT

        if not settings.SENTIMENT_MODEL_NAME:
            # 未配置模型（用户未自选）→ 关键词方案
            self.use_fallback = True
            logger.info("未配置 SENTIMENT_MODEL_NAME，直接使用关键词方案")
            return

        if settings.SENTIMENT_FALLBACK_ENABLED:
            try:
                from transformers import pipeline as hf_pipeline

                logger.info(
                    f"正在加载情感分析模型: {settings.SENTIMENT_MODEL_NAME} (镜像源: {settings.HF_ENDPOINT})"
                )
                self.pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=settings.SENTIMENT_MODEL_NAME,
                )
                logger.info("情感分析模型加载成功")
            except Exception as e:
                logger.warning(
                    f"情感分析模型加载失败: {e}，将使用关键词回退方案"
                )
                self.use_fallback = True
        else:
            self.use_fallback = True
            logger.info("已配置跳过模型加载，直接使用关键词方案")

    def analyze(self, text: str) -> Tuple[str, float]:
        """
        分析文本情感，返回 (emotion_label, emotion_score)。

        等价于 analyze_full(text) 的前两个返回值，保持既有调用方兼容。
        """
        label, score, _ = self.analyze_full(text)
        return label, score

    def analyze_full(self, text: str) -> Tuple[str, float, float]:
        """
        分析文本情感，返回 (emotion_label, emotion_score, emotion_valence)。

        逻辑：
          - 模型可用时，中英文统一走多语言模型（三分类，无需语言检测分流）
          - 模型不可用（未启用 / 加载失败）时，回退关键词方案

        两种分数的语义区分（重要）：
          - emotion_score: 标签置信度/强度（negative 0.9 = 强负面），
            供 ResponseComposer 分级关怀等按强度分层的逻辑使用
          - emotion_valence: 情绪效价（0=负面, 0.5=中性, 1=正面），
            供 MemoryAgent 情绪趋势使用——趋势的"连续 3 次 < 0.4 判负面/
            5 次均值 < 0.45 判低迷"均以效价语义设计，置信度直接入趋势
            会导致负面样本永远达不到 0.4 以下，深度关怀趋势通道失效
        """
        if not text or not text.strip():
            return "neutral", 0.5, 0.5

        if self.pipeline is not None and not self.use_fallback:
            return self._analyze_with_model(text)
        label, score = self._analyze_with_keywords(text)
        return label, score, derive_valence(label, score)

    def _analyze_with_model(self, text: str) -> Tuple[str, float, float]:
        """
        使用 transformers pipeline 进行情感分析。

        多语言模型输出 positive/neutral/negative 三分类。优先取完整分布：
          - top-1 标签 + 置信度 → (label, score)
          - 效价 = P(positive) + 0.5 × P(neutral)，充分利用三分类概率信息
        若 pipeline 不支持 top_k（mock / 旧版 transformers），退化为单结果
        + derive_valence 近似。top-1 置信度 < 0.5 视为不确定 → 中性兜底。
        """
        try:
            dist = self._model_distribution(text)
            if dist:
                top_label = max(dist, key=dist.get)
                score = round(dist[top_label], 4)
                valence = round(
                    dist.get("positive", 0.0) + 0.5 * dist.get("neutral", 0.0), 4
                )
            else:
                result = self.pipeline(text)[0]
                top_label = result["label"].lower()
                score = round(result["score"], 4)
                valence = None

            # 三分类下 top-1 置信度 < 0.5 意味着模型不确定
            # （如陈述句/疑问句被弱置信地判为 positive/negative）→ 中性兜底，
            # 避免中性内容误触发情绪关怀
            if score < 0.5:
                return "neutral", 0.5, 0.5

            if top_label == "positive":
                return "positive", score, valence if valence is not None else score
            elif top_label == "negative":
                if valence is None:
                    valence = round(1 - score, 4)
                return "negative", score, valence
            else:
                return "neutral", 0.5, 0.5
        except Exception as e:
            logger.error(f"模型情感分析异常: {e}，回退到关键词方案")
            label, score = self._analyze_with_keywords(text)
            return label, score, derive_valence(label, score)

    def _model_distribution(self, text: str):
        """
        获取三分类完整概率分布 {label: prob}。

        pipeline 不接受 top_k 参数（mock / 旧版 transformers）时
        抛 TypeError，此处返回 None 由调用方走单结果退化路径。
        """
        try:
            results = self.pipeline(text, top_k=3)
            if results and isinstance(results[0], list):
                results = results[0]  # 某些版本对单输入返回嵌套列表
            if not results or not isinstance(results, list):
                return None
            return {r["label"].lower(): float(r["score"]) for r in results}
        except TypeError:
            return None

    def _analyze_with_keywords(self, text: str) -> Tuple[str, float]:
        """
        基于关键词的本地回退情感分析。

        逻辑：
          1. 优先匹配更长的复合关键词（如"好累"优先于"累"），
             避免短词被复合词中的字干扰
          2. 统计文本中出现的正面/负面关键词数量
          3. 根据关键词命中数计算分数
          4. 若正负均无命中，判定为 neutral

        分数计算方式：
          - 命中关键词越多，分数越极端（越接近 0 或 1）
          - neutral 分数固定为 0.5
        """
        text_lower = text.lower()

        all_neg = sorted(
            self.NEGATIVE_KEYWORDS_CN + self.NEGATIVE_KEYWORDS_EN,
            key=len, reverse=True,
        )
        all_pos = sorted(
            self.POSITIVE_KEYWORDS_CN + self.POSITIVE_KEYWORDS_EN,
            key=len, reverse=True,
        )

        neg_matched = []
        temp_text = text_lower
        for kw in all_neg:
            if kw in temp_text:
                neg_matched.append(kw)
                temp_text = temp_text.replace(kw, " " * len(kw), 1)

        pos_matched = []
        temp_text = text_lower
        for kw in all_pos:
            if kw in temp_text:
                pos_matched.append(kw)
                temp_text = temp_text.replace(kw, " " * len(kw), 1)

        neg_count = len(neg_matched)
        pos_count = len(pos_matched)

        if neg_count == 0 and pos_count == 0:
            return "neutral", 0.5

        if neg_count > pos_count:
            score = min(0.95, 0.55 + neg_count * 0.08)
            return "negative", round(score, 4)
        elif pos_count > neg_count:
            score = min(0.95, 0.55 + pos_count * 0.08)
            return "positive", round(score, 4)
        else:
            return "neutral", 0.5

    def contains_code(self, text: str) -> bool:
        """
        检测文本中是否包含代码片段。
        通过常见的代码特征模式匹配：代码块标记、行尾分号、函数定义等。
        """
        code_patterns = [
            r"```[\s\S]*?```",
            r"def\s+\w+\s*\(",
            r"class\s+\w+",
            r"import\s+\w+",
            r"from\s+\w+\s+import",
            r"print\s*\(",
            r"return\s+",
            r"for\s+\w+\s+in\s+",
            r"while\s+\w+",
            r"if\s+\w+.*:",
            r"=\s*\[",
            r"\w+\.\w+\(.*\)",
        ]
        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        return False


sentiment_analyzer = SentimentAnalyzer()
