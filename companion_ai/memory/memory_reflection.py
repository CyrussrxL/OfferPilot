"""
记忆提取与反思模块 —— memory_reflection

职责：
  1. extract_facts: 对话后用 LLM 从用户消息中抽取结构化用户事实
     （学习目标、技能水平、求职进展、偏好等），存为高权重事实记忆
  2. reflect_on_profile: 周期性汇总全部事实记忆 + 现有画像，
     由 LLM 反思合并为更新后的用户画像（学习目标/技能水平/求职目标/反思摘要）

设计理由（对齐 Generative Agents 的记忆架构）：
  - 原始对话（episode）只是"经历"，直接检索会召回大量零散、含情绪噪音的原文；
    事实（fact）是对经历的"提炼"，检索时信息密度更高。
  - 反思（reflection）模拟人类"定期回顾沉淀"的行为：不逐条改画像，
    而是周期性地把累积的事实合并成连贯的用户认知，避免画像被单条对话带偏。
  - 与 DeepTravel 的"任务状态检查点"不同，本模块建模的是"用户状态"——
    这是长期陪伴型 Agent 区别于任务型 Agent 的核心。
"""

import json
import re
from typing import Dict, List

from langchain_openai import ChatOpenAI

from companion_ai.memory.vector_store import vector_store
from companion_ai.utils.config import settings
from companion_ai.utils.helpers import get_timestamp
from companion_ai.utils.logger import logger


def _get_llm() -> ChatOpenAI:
    """创建用于提取/反思的 LLM 实例（低温度保证输出稳定）。"""
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0,
        timeout=60,
        max_retries=2,
    )


def _parse_json(content: str) -> Dict:
    """宽松解析 LLM 输出的 JSON（容忍 markdown 代码块包裹）。"""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(f"输出中未找到 JSON: {content[:100]}")
    return json.loads(match.group(0))


def extract_facts(user_id: str, message: str) -> List[str]:
    """
    从用户消息中抽取结构化事实并写入事实记忆库。

    只抽取"关于用户的持久事实"，忽略：
      - 一次性提问（"这道题怎么做"）
      - 情绪表达本身（"好累啊"）
      - 寒暄（"你好"）

    与已有事实完全重复的会被跳过（去重）。

    Args:
        user_id: 用户唯一标识
        message: 用户当前消息

    Returns:
        本次新存入的事实列表
    """
    if not settings.MEMORY_FACT_EXTRACTION or not message or len(message.strip()) < 8:
        return []

    try:
        # 取已有事实用于去重
        existing_facts = vector_store.get_all_facts(user_id, limit=50)
        existing_text = "\n".join(f"- {f}" for f in existing_facts) or "（暂无）"

        prompt = f"""从下面的用户消息中抽取"关于该用户的持久事实"，用于长期用户建模。

用户消息：「{message}」

已有的用户事实（用于去重，不要重复抽取）：
{existing_text}

只抽取满足以下条件的事实：
- 关于用户的持久信息：学习目标、技能水平、求职进展（投递/面试/offer）、岗位偏好、时间安排、背景情况
- 会被"记住"而非"回答"的内容

不要抽取：一次性提问、情绪表达本身、寒暄、与用户无关的内容。
如果没有可抽取的事实，返回空列表。

只输出 JSON：
{{"facts": ["事实1（第三人称陈述，如：用户目标是字节后端岗）", ...]}}"""

        response = _get_llm().invoke(prompt)
        parsed = _parse_json(response.content.strip())
        raw_facts = parsed.get("facts", [])

        if not isinstance(raw_facts, list):
            return []

        # 去重：跳过与已有事实完全相同的新事实
        new_facts = [
            f.strip() for f in raw_facts
            if isinstance(f, str) and f.strip() and f.strip() not in existing_facts
        ]

        timestamp = get_timestamp()
        for fact in new_facts:
            vector_store.store_fact(user_id=user_id, fact=fact, timestamp=timestamp)

        if new_facts:
            logger.info(
                f"事实提取: user={user_id}, "
                f"从「{message[:30]}」提取 {len(new_facts)} 条新事实: {new_facts}"
            )
        return new_facts
    except Exception as e:
        logger.warning(f"事实提取失败（不影响主流程）: {e}")
        return []


def reflect_on_profile(user_id: str) -> Dict:
    """
    周期性反思：汇总全部事实记忆 + 现有画像，由 LLM 合成更新后的画像。

    更新字段：
      - learning_goal / current_skill_level / job_target（核心画像三元组）
      - reflection_summary: 本次反思的连贯摘要（2-3 句）
      - reflection_timestamp: 反思时间

    保留字段：emotional_trend, conversation_count, memory_compression 等
    运行时数据不受反思影响。

    Args:
        user_id: 用户唯一标识

    Returns:
        {"success": bool, "profile": 更新后的画像, "message": str}
    """
    try:
        profile = vector_store.get_user_profile(user_id)
        facts = vector_store.get_all_facts(user_id, limit=50)

        if len(facts) < 3:
            return {
                "success": False,
                "profile": profile,
                "message": f"事实记忆不足（{len(facts)} 条 < 3），跳过反思",
            }

        facts_text = "\n".join(f"- {f}" for f in facts)

        prompt = f"""你是一个长期陪伴型 AI 的反思模块。请基于累积的用户事实，更新用户画像。

当前用户画像：
{json.dumps(profile, ensure_ascii=False, indent=2)}

累积的用户事实（按时间先后，可能有过时信息，以最新为准）：
{facts_text}

请反思并输出更新后的画像：
1. 综合所有事实，更新学习目标、技能水平、求职目标（以较新的事实为准）
2. 生成 2-3 句连贯的反思摘要，描述对该用户的整体认知（学习状态、求职进展、需要注意的点）
3. 只基于事实推断，不要编造事实中没有的信息

只输出 JSON：
{{
  "learning_goal": "...",
  "current_skill_level": "...",
  "job_target": "...",
  "reflection_summary": "..."
}}"""

        response = _get_llm().invoke(prompt)
        parsed = _parse_json(response.content.strip())

        # 合并：反思结果覆盖核心画像字段，保留运行时字段
        profile["learning_goal"] = parsed.get(
            "learning_goal", profile.get("learning_goal", "")
        )
        profile["current_skill_level"] = parsed.get(
            "current_skill_level", profile.get("current_skill_level", "")
        )
        profile["job_target"] = parsed.get(
            "job_target", profile.get("job_target", "")
        )
        profile["reflection_summary"] = parsed.get("reflection_summary", "")
        profile["reflection_timestamp"] = get_timestamp()

        vector_store.save_user_profile(user_id, profile)

        logger.info(
            f"画像反思完成: user={user_id}, "
            f"基于 {len(facts)} 条事实, "
            f"summary={profile['reflection_summary'][:60]}"
        )
        return {"success": True, "profile": profile, "message": "反思完成"}
    except Exception as e:
        logger.warning(f"画像反思失败（不影响主流程）: {e}")
        return {"success": False, "profile": {}, "message": str(e)}
