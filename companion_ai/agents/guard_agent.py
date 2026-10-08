"""
GuardAgent —— 入口 Agent

职责：
  1. 接收用户消息，调用情感分析模块获取 emotion_label 和 emotion_score
  2. 判断消息类别（coding/career/emotional/chitchat）
  3. 将情感标签、分数和消息类别附加到状态中，供后续 Agent 使用

设计理由：
  - 作为工作流的入口节点，GuardAgent 是所有消息的第一站。
  - 情感分析前置，使得后续所有 Agent 都能感知用户情绪，
    为情绪驱动的回复风格切换提供基础。
  - 消息分类决定了条件路由的走向，是 LangGraph 条件边的核心依据。
  - 分类逻辑使用向量相似度检测，结合代码检测和关键词兜底，兼顾智能性和鲁棒性。
"""

from typing import Dict, Optional

from companion_ai.agents.behavior_analyzer import behavior_analyzer
from companion_ai.emotion.sentiment_analyzer import sentiment_analyzer
from companion_ai.graph.state import State
from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


VALID_CATEGORIES = ("coding", "career", "emotional", "chitchat")


CAREER_KEYWORDS = [
    "实习", "简历", "面试", "offer", "求职", "招聘", "投递",
    "秋招", "春招", "内推", "笔试", "HR", "薪资", "转正",
    "海康", "大疆", "华为", "禾赛", "字节", "腾讯", "阿里",
    "美团", "百度", "小米", "网易", "京东", "快手",
    "岗位", "公司", "工作", "职业", "就业",
]

CODING_KEYWORDS = [
    "python", "代码", "算法", "bug", "调试", "leetcode", "力扣",
    "函数", "类", "递归", "排序", "二叉树", "动态规划", "dp",
    "链表", "数组", "哈希", "栈", "队列", "图", "dfs", "bfs",
    "报错", "error", "exception", "traceback", "debug",
    "编程", "程序", "变量", "循环", "条件", "继承", "多态",
    "数据结构", "时间复杂度", "空间复杂度",
]

EMOTIONAL_KEYWORDS = [
    "焦虑", "累", "烦", "难过", "崩溃", "压力", "抑郁", "沮丧",
    "开心", "谢谢", "心情", "情绪", "烦死了", "受不了",
    "想哭", "好累", "太难了", "不想学了", "迷茫",
]

# 强意图词：单独命中一次即视为存在该意图（弱词需命中 >= 2 次降噪）。
# 背景：混合消息常被主类别淹没，如"我很开心，帮我做求职规划"被分类为
# emotional 时，"求职"仅命中 1 次，若按弱词门槛会漏掉用户的求职需求。
CAREER_STRONG_KEYWORDS = [
    "求职", "简历", "秋招", "春招", "offer", "内推", "投递",
    "面试", "职业规划", "笔试",
]
CODING_STRONG_KEYWORDS = [
    "代码", "算法", "报错", "error", "exception", "traceback",
    "bug", "调试", "leetcode", "力扣", "动态规划", "数据结构",
]


def _classify_message_with_keywords(text: str) -> tuple:
    """
    基于关键词匹配对消息进行分类（兜底方案）。

    优先级：coding > career > emotional > chitchat
    理由：编程问题通常包含代码特征，优先级最高；
    求职问题关键词明确，次之；情绪问题需要关注，但可由 EmotionAgent 补充；
    其余归为闲聊。

    Returns:
        (category, confidence) 类别和置信度
    """
    text_lower = text.lower()

    coding_hits = sum(1 for kw in CODING_KEYWORDS if kw in text_lower)
    career_hits = sum(1 for kw in CAREER_KEYWORDS if kw in text_lower)
    emotional_hits = sum(1 for kw in EMOTIONAL_KEYWORDS if kw in text_lower)

    scores = {
        "coding": coding_hits,
        "career": career_hits,
        "emotional": emotional_hits,
    }

    max_category = max(scores, key=scores.get)
    max_score = scores[max_category]

    if max_score == 0:
        return "chitchat", 0.5

    # 计算置信度：最高分占总分的比例
    total_score = sum(scores.values())
    confidence = max_score / total_score if total_score > 0 else 0.5

    return max_category, min(confidence, 1.0)


def _has_task_intent(text_lower: str, strong_kws: list, all_kws: list) -> bool:
    """判断消息中是否存在某类任务意图：强词命中 1 次或弱词命中 >= 2 次。"""
    if any(kw in text_lower for kw in strong_kws):
        return True
    return sum(1 for kw in all_kws if kw in text_lower) >= 2


def _detect_secondary_category(
    text: str, primary_category: str
) -> str:
    """
    次意图检测：识别混合意图消息中主类别之外的明确意图。

    场景：用户常在一条消息里同时表达多个需求，如
    "今天拿到 offer 了特别开心，帮我做下求职规划"（career 主 + emotional 次）、
    "最近好焦虑，顺便帮我看看简历怎么改"（emotional 主 + career 次）。
    单标签路由只能回应主意图，次意图信息会丢失——这里用关键词命中
    做轻量检测（零 LLM 成本），结果由 ResponseComposer 追加回应。

    规则（强意图词单次命中即触发；弱词命中 >= 2 次降噪）：
      - 主类别为任务型(coding/career)：
          * 情绪词命中 >= 1 → 次意图 emotional
          * coding/career 互检：对方存在任务意图（强词或弱词>=2）→ 次意图为对方类别
      - 主类别为 emotional/chitchat：
          * 存在 coding 或 career 任务意图 → 次意图为对应任务类别

    Returns:
        次意图类别（VALID_CATEGORIES 之一），无次意图时为空字符串
    """
    text_lower = text.lower()

    emotional_hits = sum(1 for kw in EMOTIONAL_KEYWORDS if kw in text_lower)
    coding_intent = _has_task_intent(
        text_lower, CODING_STRONG_KEYWORDS, CODING_KEYWORDS
    )
    career_intent = _has_task_intent(
        text_lower, CAREER_STRONG_KEYWORDS, CAREER_KEYWORDS
    )

    if primary_category in ("coding", "career"):
        if emotional_hits >= 1:
            return "emotional"
        if primary_category == "coding" and career_intent:
            return "career"
        if primary_category == "career" and coding_intent:
            return "coding"
        return ""

    if primary_category in ("emotional", "chitchat"):
        if coding_intent:
            return "coding"
        if career_intent:
            return "career"

    return ""


def _llm_arbitrate(
    text: str,
    vector_category: str,
    vector_confidence: float,
) -> Optional[tuple]:
    """
    LLM few-shot 仲裁：低置信度消息的最终裁决层。

    将向量分类检索到的 top-3 相似种子作为 few-shot 示例，
    连同向量分类结果一起交给 LLM 做语义级判断。

    设计理由：
      - 向量分类在语义开放集上会产出低置信度结果（如"面试官问了我一道 DP 题"
        同时贴近 career 和 coding），此时需要 LLM 的语义推理能力做仲裁。
      - 与 DeepTravel 的"LLM 不做路由"哲学相反：陪伴场景的语义空间是开放的，
      纯代码规则会误伤，因此低置信度分支刻意引入 LLM 决策。

    Args:
        text: 用户消息
        vector_category: 向量分类的类别
        vector_confidence: 向量分类的置信度

    Returns:
        (category, confidence) 或 None（LLM 仲裁失败）
    """
    try:
        from langchain_openai import ChatOpenAI

        from companion_ai.memory.vector_store import vector_store

        # 取 top-3 相似种子作为 few-shot 示例
        seeds = vector_store.get_similar_seeds(text, top_k=3)
        if not seeds:
            return None

        examples = "\n".join(
            f"  「{s['text']}」 -> {s['category']}" for s in seeds
        )

        prompt = f"""你是消息分类仲裁器。向量分类器对下面这条消息给出的结果置信度偏低，请结合相似示例做最终裁决。

类别定义：
- coding: 编程/算法/代码调试/技术学习
- career: 求职/简历/面试/offer/职业规划
- emotional: 情绪倾诉/压力焦虑/寻求安慰
- chitchat: 日常闲聊/打招呼/无关话题

用户消息：「{text}」

最相似的已标注示例：
{examples}

向量分类器的结果：{vector_category}（置信度 {vector_confidence:.2f}）

注意：示例仅供参考，请基于消息本身的核心意图判断，向量结果可能是错的。

只输出 JSON，不要输出其他内容：
{{"category": "coding/career/emotional/chitchat 之一", "reason": "一句话理由"}}"""

        llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0,
            timeout=30,
            max_retries=1,
        )
        response = llm.invoke(prompt)
        content = response.content.strip()

        # 宽松解析 JSON（容忍 markdown 代码块包裹）
        import json as _json
        import re as _re

        match = _re.search(r"\{.*\}", content, _re.DOTALL)
        if not match:
            logger.warning(f"LLM 仲裁输出无法解析: {content[:100]}")
            return None

        parsed = _json.loads(match.group(0))
        category = parsed.get("category", "").strip()
        if category not in VALID_CATEGORIES:
            logger.warning(f"LLM 仲裁返回非法类别: {category}")
            return None

        # 仲裁结果的置信度：高于触发阈值，但保留对向量结果的尊重
        # （若仲裁与向量结论一致则更高，不一致取仲裁值 0.75）
        if category == vector_category:
            arbitrated_confidence = max(vector_confidence, 0.8)
        else:
            arbitrated_confidence = 0.75

        logger.info(
            f"LLM 仲裁: {vector_category}({vector_confidence:.2f}) -> "
            f"{category}({arbitrated_confidence:.2f}), "
            f"reason={parsed.get('reason', '')[:50]}"
        )
        return category, arbitrated_confidence
    except Exception as e:
        logger.warning(f"LLM 仲裁失败，沿用向量分类结果: {e}")
        return None


def _classify_message(text: str, behavior: Dict = None) -> tuple:
    """
    置信度分级的消息分类（主方案）。

    分层策略：
      1. 代码检测（最高优先级）→ 直接 coding
      2. 向量相似度分类：
         - 置信度 >= 阈值(0.6) → 直接采用（高置信直出）
         - 置信度 < 阈值 → LLM few-shot 仲裁（低置信仲裁），失败则沿用向量结果
      3. 行为特征修正（仲裁后仍 <0.8 时叠加）
      4. 关键词兜底（向量分类整体异常时）

    Args:
        text: 用户消息
        behavior: 用户行为特征（可选）

    Returns:
        (category, confidence, arbitration_used) 类别、置信度、是否经过 LLM 仲裁
    """
    # 1. 代码检测（最高优先级）
    if sentiment_analyzer.contains_code(text):
        return "coding", 0.95, False

    # 2. 向量相似度分类（主方案）
    vector_category, vector_confidence = "chitchat", 0.5
    vector_ok = False
    arbitration_used = False
    try:
        from companion_ai.memory.vector_store import vector_store
        vector_category, vector_confidence = (
            vector_store.classify_message_with_confidence(text, top_k=3)
        )
        vector_ok = True

        # 2a. 低置信度 → LLM few-shot 仲裁
        if (
            settings.GUARD_LLM_ARBITRATION
            and vector_confidence < settings.GUARD_ARBITRATION_THRESHOLD
        ):
            arbitrated = _llm_arbitrate(text, vector_category, vector_confidence)
            if arbitrated is not None:
                vector_category, vector_confidence = arbitrated
                arbitration_used = True
    except Exception as e:
        logger.warning(f"向量分类失败，回退到关键词方案: {e}")

    # 3. 行为特征调整（多模态分类）
    if vector_ok and behavior and vector_confidence < 0.8:
        vector_category, vector_confidence = _adjust_category_with_behavior(
            vector_category, vector_confidence, text, behavior
        )

    if vector_ok:
        return vector_category, vector_confidence, arbitration_used

    # 4. 关键词兜底
    category, confidence = _classify_message_with_keywords(text)
    return category, confidence, False


def _adjust_category_with_behavior(
    category: str,
    confidence: float,
    text: str,
    behavior: Dict,
) -> tuple:
    """
    根据行为特征调整分类结果。

    规则：
      - 快速短消息 + 深夜 → 可能情绪化
      - 情绪化倾向高 → 提升 emotional 类别权重
      - 长消息 + 慢速输入 → 可能深思熟虑，保持原分类
    """
    emotional_tendency = behavior.get("emotional_tendency", "neutral")
    is_late_night = behavior.get("is_late_night", False)
    typing_speed = behavior.get("typing_speed", "unknown")
    message_length = behavior.get("message_length_category", "unknown")

    # 规则 1：快速短消息 + 深夜 → 可能情绪化
    if emotional_tendency == "likely_emotional" and category in ("chitchat", "career"):
        # 检查是否包含情绪相关词汇
        text_lower = text.lower()
        emotional_indicators = ["唉", "哎", "烦", "累", "难", "烦死了", "受不了"]
        if any(ind in text_lower for ind in emotional_indicators):
            logger.info(f"行为特征调整分类: {category} -> emotional (深夜/快速短消息)")
            return "emotional", confidence + 0.1

    # 规则 2：深思熟虑型 → 保持原分类，提升置信度
    if emotional_tendency == "likely_thoughtful":
        logger.info(f"行为特征确认分类: {category} (深思熟虑型)")
        return category, min(confidence + 0.05, 1.0)

    # 规则 3：深夜 + 求职相关 → 可能焦虑
    if is_late_night and category == "career" and typing_speed == "fast":
        logger.info(f"行为特征调整分类: {category} -> emotional (深夜求职焦虑)")
        return "emotional", confidence + 0.15

    return category, confidence


def record_classification_correction(
    user_id: str, message: str, correct_category: str
) -> Dict:
    """
    记录分类纠正（错分反馈闭环入口）。

    用户发现 GuardAgent 路由错分后调用此接口，将 (消息, 正确类别)
    作为新种子回流到分类种子库，后续同类消息的向量分类即可命中，
    实现分类系统的自我改进。

    Args:
        user_id: 用户唯一标识
        message: 被错分的消息原文
        correct_category: 用户纠正后的正确类别

    Returns:
        {"success": bool, "message": str, "seed_id": str}
    """
    from companion_ai.memory.vector_store import vector_store

    if not message or not message.strip():
        return {"success": False, "message": "消息不能为空", "seed_id": ""}

    result = vector_store.add_classification_seed(
        text=message.strip(),
        category=correct_category,
        source="feedback",
    )
    if result.get("success"):
        logger.info(
            f"错分反馈闭环: user={user_id}, "
            f"「{message[:30]}」已回流为 {correct_category} 种子"
        )
    return result


def guard_agent(state: State) -> Dict:
    """
    GuardAgent 节点函数。

    流程：
      1. 从状态中获取用户消息
      2. 分析用户行为特征（输入频率、消息长度、时间段）
      3. 调用情感分析器获取 emotion_label 和 emotion_score
      4. 对消息进行置信度分级分类（代码检测/向量/LLM仲裁/行为修正）
      5. 记录分类反馈（含仲裁标记，供错分闭环分析）
      6. 返回更新后的状态字段

    Args:
        state: 当前 LangGraph 状态

    Returns:
        包含 emotion_label, emotion_score, message_category, user_behavior,
        classification_confidence 的状态更新字典
    """
    message = state.get("current_message", "")
    user_id = state.get("user_id", "default_user")

    # 1. 分析行为特征（多模态分类）
    behavior = behavior_analyzer.analyze_behavior(user_id, message)

    # 2. 情感分析
    emotion_label, emotion_score = sentiment_analyzer.analyze(message)

    # 3. 消息分类（置信度分级：高置信直出 / 低置信 LLM 仲裁）
    message_category, classification_confidence, arbitration_used = (
        _classify_message(message, behavior)
    )

    # 3b. 次意图检测（混合意图消息的主类别之外意图，供 ResponseComposer 追加回应）
    secondary_category = _detect_secondary_category(message, message_category)

    # 4. 记录分类反馈
    classification_feedback = {
        "user_id": user_id,
        "message": message[:50],
        "category": message_category,
        "secondary_category": secondary_category,
        "confidence": classification_confidence,
        "arbitration_used": arbitration_used,
        "behavior": behavior,
        "timestamp": behavior.get("timestamp", ""),
    }

    logger.info(
        f"GuardAgent | user={user_id} | "
        f"emotion={emotion_label}({emotion_score:.2f}) | "
        f"category={message_category} (confidence={classification_confidence:.2f}, "
        f"arbitration={arbitration_used}) | "
        f"secondary={secondary_category or '无'} | "
        f"behavior={behavior.get('emotional_tendency', 'unknown')} | "
        f"message={message[:50]}..."
    )

    return {
        "emotion_label": emotion_label,
        "emotion_score": emotion_score,
        "message_category": message_category,
        "secondary_category": secondary_category,
        "user_behavior": behavior,
        "classification_confidence": classification_confidence,
        "classification_feedback": [classification_feedback],
    }
