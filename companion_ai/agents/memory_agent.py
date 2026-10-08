"""
MemoryAgent —— 记忆 Agent

职责：
  1. 双路召回：事实记忆（LLM 提取的结构化用户事实，高权重）
     + 原文记忆（历史对话，带时间衰减/频率加权）
  2. 主动记忆检索（根据情绪状态推送相关记忆）
  3. 获取用户画像
  4. 将当前消息和情感分析结果存入向量库
  5. 对话后事实提取（LLM 抽取结构化用户事实）
  6. 更新用户画像中的情绪趋势
  7. 周期性画像反思（每 N 次对话，LLM 合并事实更新画像）

设计理由：
  - "事实 + 原文"双路召回：事实记忆信息密度高、无情绪噪音，
    原文记忆保留对话细节，两路互补（对齐 Generative Agents）。
  - 事实提取让记忆从"存经历"升级为"存认知"，检索质量随对话累积提升。
  - 周期性反思模拟人类"定期回顾沉淀"，画像更新平滑而非被单条对话带偏。
"""

from typing import Dict

from companion_ai.emotion.sentiment_analyzer import derive_valence
from companion_ai.graph.state import State
from companion_ai.memory.memory_reflection import extract_facts, reflect_on_profile
from companion_ai.memory.vector_store import vector_store
from companion_ai.utils.config import settings
from companion_ai.utils.helpers import get_timestamp
from companion_ai.utils.logger import logger


def memory_agent(state: State) -> Dict:
    """
    MemoryAgent 节点函数。

    流程：
      1. 双路召回：事实记忆（top-2）+ 原文记忆（top-3，权重衰减）
      2. 主动记忆检索（根据情绪状态推送相关记忆）
      3. 获取用户画像
      4. 将当前消息存入向量库（episode）
      5. 事实提取：LLM 从当前消息抽取结构化用户事实（fact）
      6. 更新情绪趋势
      7. 检查记忆压缩 + 周期性画像反思
      8. 返回更新后的状态字段

    Args:
        state: 当前 LangGraph 状态

    Returns:
        包含 retrieved_memories, proactive_memories, user_profile 的状态更新字典
    """
    user_id = state.get("user_id", "default_user")
    message = state.get("current_message", "")
    emotion_label = state.get("emotion_label", "neutral")
    emotion_score = state.get("emotion_score", 0.5)
    # 情绪效价（0=负面, 0.5=中性, 1=正面）：趋势与主动记忆检索均按效价
    # 语义设计；旧状态无此字段（如历史检查点恢复）时由 label/score 推导
    emotion_valence = state.get("emotion_valence")
    if emotion_valence is None:
        emotion_valence = derive_valence(emotion_label, emotion_score)
    message_category = state.get("message_category", "chitchat")

    # 1a. 事实记忆召回（高权重、信息密度高）
    fact_memories = vector_store.retrieve_facts(
        user_id=user_id,
        query=message,
        top_k=2,
    )

    # 1b. 原文记忆召回（历史对话，应用权重衰减）
    episode_memories = vector_store.retrieve_memories(
        user_id=user_id,
        query=message,
        top_k=3,
        apply_decay=True,
    )

    # 合并双路结果：事实在前（权重加成后通常得分更高）
    retrieved_memories = fact_memories + episode_memories

    # 2. 主动记忆检索（根据情绪状态推送相关记忆，效价语义）
    proactive_memories = vector_store.proactive_memory_retrieval(
        user_id=user_id,
        current_emotion=emotion_label,
        emotion_score=emotion_valence,
        top_k=2,
    )

    # 3. 获取用户画像
    user_profile = vector_store.get_user_profile(user_id)

    # 4. 将当前消息存入向量库（episode 原文记忆）
    timestamp = get_timestamp()
    vector_store.store_conversation(
        user_id=user_id,
        text=message,
        emotion=emotion_label,
        category=message_category,
        timestamp=timestamp,
        role="user",
        memory_type="episode",
    )

    # 5. 事实提取：LLM 抽取结构化用户事实（fact 记忆）
    extract_facts(user_id=user_id, message=message)

    # 6. 更新对话计数（先保存，避免覆盖下一步写入的情绪趋势：
    #    user_profile 是本轮开头取的快照，若在 update_emotional_trend
    #    之后保存，会把刚追加的情绪效价抹掉）
    conversation_count = user_profile.get("conversation_count", 0) + 1
    user_profile["conversation_count"] = conversation_count
    vector_store.save_user_profile(user_id, user_profile)

    # 7. 更新情绪趋势（效价语义：0=负面, 0.5=中性, 1=正面；置信度直接
    #    入趋势会让负面样本永远 ≥0.55，趋势通道的深度关怀永不触发）。
    #    内部重新读取最新画像后追加落库；同时同步回本地快照——下游节点
    #    （如 CareerAgent）会用 state 中的画像回写存储，若快照缺最新趋势，
    #    回写时会把趋势覆盖掉
    user_profile["emotional_trend"] = vector_store.update_emotional_trend(
        user_id, emotion_valence
    )

    # 8. 记忆容量治理（每 10 次对话检查一次）
    #    每轮写 2 条 episode（用户消息 + AI 回复），200 条 ≈ 100 轮后首次触发，
    #    压缩后保留最近 100 条（≈ 50 轮完整对话），fact 记忆不受影响
    if conversation_count % 10 == 0:
        compression_result = vector_store.compress_memories(
            user_id=user_id,
            max_memories=200,
            keep_recent=100,
        )
        logger.info(f"记忆压缩检查: {compression_result.get('status', 'unknown')}")

    # 9. 周期性画像反思（每 N 次对话触发一次，LLM 合并事实更新画像）
    if conversation_count % settings.MEMORY_REFLECTION_INTERVAL == 0:
        reflection_result = reflect_on_profile(user_id)
        if reflection_result.get("success"):
            user_profile = reflection_result["profile"]
        logger.info(
            f"画像反思: user={user_id}, "
            f"result={reflection_result.get('message', 'unknown')}"
        )

    logger.info(
        f"MemoryAgent | user={user_id} | "
        f"facts={len(fact_memories)} | episodes={len(episode_memories)} | "
        f"proactive={len(proactive_memories)} | "
        f"profile_keys={list(user_profile.keys())}"
    )

    return {
        "retrieved_memories": retrieved_memories,
        "proactive_memories": proactive_memories,
        "user_profile": user_profile,
    }
