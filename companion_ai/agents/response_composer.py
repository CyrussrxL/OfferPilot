"""
ResponseComposer —— 响应合成器节点

职责：
  1. 将各 Agent 的回复（coding_response / career_response / general_response）
     与情绪关怀语句拼接成最终的 final_response
  2. 将 assistant 的回复存入记忆，保持对话历史的完整性
  3. 支持日报生成功能（长期记忆的总结能力）

设计理由：
  - ResponseComposer 作为工作流的最终节点，确保所有路径的输出都经过统一处理。
  - 情绪关怀语句前置插入，让用户第一时间感受到 AI 的关心。
  - 将情绪关怀逻辑从独立 Agent 合并到此节点，避免过度设计，
    同时保留基于历史趋势的分级关怀能力。
"""

from typing import Dict

from langchain_openai import ChatOpenAI

from companion_ai.graph.state import State
from companion_ai.memory.vector_store import vector_store
from companion_ai.utils.config import settings
from companion_ai.utils.helpers import get_timestamp, truncate_text
from companion_ai.utils.logger import logger


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0.7,
        timeout=120,
        max_retries=3,
    )


# 分级关怀模板：同一层级多个变体轮换。
# 动机：连续多轮负面时用户会收到一字不差的重复关怀语，
# 模板感会削弱真诚感；轮换让固定文案的"可复用"与"不可预测"并存。
# 约束：同层级的所有变体保留相同的层级 emoji，保证分级语义稳定
# （前端/测试均以 emoji 锚定层级）。
_CARE_TEMPLATES = {
    "deep": [
        "💙 我注意到你最近有些焦虑和疲惫，要不下次试试先深呼吸？"
        "我一直在这里陪着你，不用急着赶路，照顾好自己才是最重要的。",
        "💙 最近的状态我都看在眼里，别对自己太苛刻了。"
        "走得慢一点没关系，我一直陪着你。",
        "💙 连着几条消息都能感觉到你的低落。今晚早点休息吧，"
        "明天的事明天再说，我哪儿也不去。",
    ],
    "negative_high": [
        "🤗 感觉你现在的情绪波动比较大，先别急，"
        "深呼吸一下，慢慢来，我在这里陪你。",
        "🤗 我能感受到你现在心里不太平静。"
        "先停十秒，我们一起慢慢把事情捋清楚。",
    ],
    "negative_mid": [
        "😊 看起来你有些不太开心，没关系，"
        "每个人都会有低谷的时候，休息一下也许会好一些。",
        "😊 有点低落也很正常，不用强撑着，想说什么都可以跟我说。",
    ],
    "negative_low": [
        "🫂 稍微有点低落？没关系的，我在呢。",
        "🫂 有点不开心呀，先给你递杯热水，慢慢说。",
    ],
    "positive_high": [
        "🌟 看到你状态这么好真开心！继续保持这份热情！",
        "🌟 这份好状态太难得啦！乘胜追击，也别忘了给自己留点余量～",
    ],
    "positive_mid": [
        "👍 不错的心情！希望你能一直保持积极的状态～",
        "👍 心情在线的一天！稳稳地继续保持～",
    ],
    "neutral_coding": [
        "📝 来学习啦？加油！",
        "📝 又到刷题时间了？我陪你一起啃。",
    ],
    "neutral_career": [
        "💼 求职路上有我陪你！",
        "💼 一步一步来，你的每份准备我都看在眼里。",
    ],
    "neutral_default": [
        "👋 你好呀～",
        "👋 在呢，随时想聊就聊～",
    ],
}

# 各层级轮换指针（模块级）
_care_rotation: Dict[str, int] = {}


def _pick_care(tier: str) -> str:
    """轮换取关怀语：同层级连续触发时按序换变体，绕一圈回到第一条。"""
    variants = _CARE_TEMPLATES[tier]
    idx = _care_rotation.get(tier, 0) % len(variants)
    _care_rotation[tier] = idx + 1
    return variants[idx]


def _reset_care_rotation() -> None:
    """重置轮换指针（测试用，保证断言确定性）。"""
    _care_rotation.clear()


def _generate_emotion_care(
    emotion_label: str,
    emotion_score: float,
    emotional_trend: list,
    message_category: str,
) -> str:
    """
    根据情绪状态和历史趋势生成关怀语句。

    逻辑：
      1. 连续负面：趋势末尾连续 3 次 < 0.4 → 深度关怀
      2. 持续低迷：趋势窗口（最近 5 次）均值 < 0.45 且样本 ≥ 4 → 深度关怀。
         捕捉"正负交替但整体低位"的慢性低落——这类用户永远凑不齐
         3 连低，但同样需要被看见
      3. 按当前情绪标签和分数分级关怀
      4. 结合消息类别调整关怀内容
      5. 同层级模板轮换，避免连续触发时的重复感
    """
    consecutive_negative = 0
    for score in reversed(emotional_trend):
        if score < 0.4:
            consecutive_negative += 1
        else:
            break

    # 持续低迷：滑动窗口均值（窗口与 update_emotional_trend 的 5 条对齐）
    window = emotional_trend[-5:]
    sustained_low = len(window) >= 4 and sum(window) / len(window) < 0.45

    if consecutive_negative >= 3 or sustained_low:
        return _pick_care("deep")

    if emotion_label == "negative":
        if emotion_score >= 0.7:
            return _pick_care("negative_high")
        elif emotion_score >= 0.4:
            return _pick_care("negative_mid")
        else:
            return _pick_care("negative_low")

    elif emotion_label == "positive":
        if emotion_score >= 0.7:
            return _pick_care("positive_high")
        else:
            return _pick_care("positive_mid")

    else:
        if message_category == "coding":
            return _pick_care("neutral_coding")
        elif message_category == "career":
            return _pick_care("neutral_career")
        else:
            return _pick_care("neutral_default")


# 次意图（secondary_category）引导文案：主回复后的追加段
_SECONDARY_HINTS = {
    "career": (
        "📋 顺便看到你还提到了求职相关的事——想深入聊聊的话，"
        "把简历或目标岗位发我，我帮你做技能 Gap 分析和求职规划。"
    ),
    "coding": (
        "💻 你提到的技术问题我也可以帮忙——把代码或题目发过来，"
        "我带你一步步看。"
    ),
}


def _generate_secondary_response(
    secondary_category: str,
    primary_category: str,
    emotion_label: str,
) -> str:
    """
    生成次意图回应段（追加在主回复之后）。

    拼接规则：
      - secondary == emotional 且主类别为任务型：
          * 当前情绪为 neutral 时，前置情绪关怀不会触发（composer 只在
            positive/negative 时前置），这里补一段轻量情绪呼应；
          * positive/negative 时已有前置关怀，跳过避免重复。
      - secondary == coding/career（主类别为情绪/闲聊/另一任务）：
          追加任务引导段，邀请用户展开该意图。
    """
    if not secondary_category or secondary_category == primary_category:
        return ""

    if secondary_category == "emotional":
        if primary_category in ("coding", "career") and emotion_label == "neutral":
            return "💙 另外也谢谢你和我分享心情，我都记在心里。"
        return ""

    return _SECONDARY_HINTS.get(secondary_category, "")


def _build_general_prompt(state: State) -> str:
    """
    构建 general_chat 节点的提示词，用于处理 emotional 和 chitchat 类别的消息。

    注意：情绪关怀（鼓励安慰）统一由 ResponseComposer 前置处理，
    此 Prompt 只注入情绪原始值供 Agent 参考，不重复注入风格指令。
    """
    from companion_ai.utils.helpers import format_memories, format_user_profile

    message = state.get("current_message", "")
    emotion_label = state.get("emotion_label", "neutral")
    emotion_score = state.get("emotion_score", 0.5)
    memories = state.get("retrieved_memories", [])
    profile = state.get("user_profile", {})

    memories_text = format_memories(memories)
    profile_text = format_user_profile(profile)

    prompt = f"""你是一位温暖、智能的求职领航员，名叫 OfferPilot。

## 用户画像
{profile_text}

## 相关历史记忆
{memories_text}

## 当前情绪状态（仅供参考，共情能力自然发挥即可）
情绪标签: {emotion_label}, 情绪分数: {emotion_score:.2f}

## 用户消息
{message}

请给出温暖、有深度的回复。"""

    return prompt


def general_chat(state: State) -> Dict:
    """
    general_chat 节点函数，处理 emotional 和 chitchat 类别的消息。
    使用大模型通用回复，但需携带记忆和情绪感知。
    """
    user_id = state.get("user_id", "default_user")
    logger.info(f"GeneralChat | user={user_id} | 处理通用对话")

    prompt = _build_general_prompt(state)

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        general_response = response.content
    except Exception as e:
        logger.error(f"GeneralChat | LLM 调用失败: {e}")
        general_response = "抱歉，我暂时无法回复，请稍后再试。"

    return {"general_response": general_response}


def response_composer(state: State) -> Dict:
    """
    ResponseComposer 节点函数。

    流程：
      1. 根据情绪状态和历史趋势生成关怀语句
      2. 获取各 Agent 的专业回复
      3. 拼接为 final_response（情绪关怀前置）
      4. 将 assistant 回复存入记忆

    拼接格式：
      [情绪关怀语句]

      ---

      [主要回复内容]
    """
    user_id = state.get("user_id", "default_user")
    emotion_label = state.get("emotion_label", "neutral")
    emotion_score = state.get("emotion_score", 0.5)
    message_category = state.get("message_category", "chitchat")
    user_profile = state.get("user_profile", {})

    # 1. 生成情绪关怀语句
    emotional_trend = user_profile.get("emotional_trend", [])
    emotion_care = _generate_emotion_care(
        emotion_label=emotion_label,
        emotion_score=emotion_score,
        emotional_trend=emotional_trend,
        message_category=message_category,
    )

    # 2. 获取专业回复（面试模式下优先使用面试状态机输出）
    if state.get("interview_mode") and state.get("interview_response"):
        main_response = state.get("interview_response", "")
    elif message_category == "coding":
        main_response = state.get("coding_response", "")
    elif message_category == "career":
        main_response = state.get("career_response", "")
    else:
        main_response = state.get("general_response", "")

    if not main_response:
        main_response = "我暂时无法处理这个请求，请稍后再试。"

    # 3. 拼接最终回复
    if emotion_care and emotion_label != "neutral":
        final_response = f"{emotion_care}\n\n---\n\n{main_response}"
    else:
        final_response = main_response

    # 3b. 次意图回应段（混合意图消息：主回复后追加对次意图的回应/引导）
    secondary_response = _generate_secondary_response(
        secondary_category=state.get("secondary_category", ""),
        primary_category=message_category,
        emotion_label=emotion_label,
    )
    if secondary_response:
        final_response = f"{final_response}\n\n---\n\n{secondary_response}"

    # 4. 存储到记忆（截断至 500 字：AI 全文回复可达 3000+ 字，
    #    全文入库是记忆存储与检索噪音的主要来源；前端完整历史
    #    另存 conversations/*.json，UI 展示不受影响）
    timestamp = get_timestamp()
    vector_store.store_conversation(
        user_id=user_id,
        text=truncate_text(final_response, 500),
        emotion=emotion_label,
        category=message_category,
        timestamp=timestamp,
        role="assistant",
    )

    logger.info(
        f"ResponseComposer | user={user_id} | "
        f"category={message_category} | "
        f"emotion={emotion_label} | "
        f"final_response_len={len(final_response)}"
    )

    return {"final_response": final_response}


def generate_daily_report(state: State) -> Dict:
    """
    深度加分项：日报生成功能。
    根据用户今天的学习/情绪情况生成日报摘要。
    """
    user_id = state.get("user_id", "default_user")

    summary_text = vector_store.generate_summary(user_id, "今天的学习和情绪情况")

    prompt = f"""请根据以下用户数据，生成一份今日学习与情绪日报：

{summary_text}

日报格式：
1. 今日学习总结
2. 情绪状态分析
3. 明日建议

请用温暖、鼓励的语气撰写。"""

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        daily_report = response.content
    except Exception as e:
        logger.error(f"日报生成失败: {e}")
        daily_report = f"日报生成失败: {str(e)}"

    logger.info(f"DailyReport | user={user_id} | 日报生成完成")

    return {"daily_report": daily_report}
