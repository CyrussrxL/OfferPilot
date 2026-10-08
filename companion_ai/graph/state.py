"""
LangGraph 状态定义 —— State

职责：
  定义多 Agent 协作中传递的共享状态对象。
  所有 Agent 节点通过读写此状态进行通信，实现解耦。

设计理由：
  - 使用 TypedDict 定义状态结构，LangGraph 要求状态为 dict-like 对象。
  - 每个字段对应工作流中某个阶段的数据产出：
    - guard 阶段产出: emotion_label, emotion_score, message_category
    - memory 阶段产出: retrieved_memories, user_profile
    - coding/career/general_chat 阶段产出: coding_response / career_response / general_response
    - response_composer 阶段产出: final_response（整合情绪关怀与专业回复）
  - 这种设计让每个 Agent 只关注自己需要读取和写入的字段，
    避免了 Agent 之间的直接依赖。
"""

from typing import Any, Dict, List, Optional, TypedDict


class State(TypedDict, total=False):
    """
    多 Agent 协作的共享状态。

    字段说明：
      user_id: 用户唯一标识，用于记忆检索和画像管理
      current_message: 用户当前输入的消息
      emotion_label: 情感标签 (positive/negative/neutral)，由 GuardAgent 产出
      emotion_score: 情感分数 (0.0~1.0)，由 GuardAgent 产出
      message_category: 消息类别 (coding/career/emotional/chitchat)，由 GuardAgent 产出
      secondary_category: 次要意图类别（主类别之外明确存在的另一意图，如"情绪+求职"
        混合消息中的情绪意图），无次意图时为空字符串，由 GuardAgent 产出
      retrieved_memories: 检索到的相关历史记忆列表，由 MemoryAgent 产出
      proactive_memories: 主动推送的相关记忆列表，由 MemoryAgent 产出
      user_profile: 用户画像字典，由 MemoryAgent 产出
      coding_response: 编程辅导回复，由 CodingAgent 产出
      career_response: 求职建议回复，由 CareerAgent 产出
      general_response: 通用回复，由 general_chat 节点产出
      final_response: 最终拼接的完整回复（含情绪关怀），由 ResponseComposer 产出
      interview_mode: 是否处于模拟面试模式
      interview_question: 当前面试问题
      interview_score: 面试评分
      interview_session: 循环面试状态机的会话状态（跨轮经检查点持久化），
        包含 questions（已问题目/回答/评分）、awaiting_answer、asked_count、
        probe_count 等字段，由 interview_evaluate/interview_ask 节点维护
      interview_response: 面试官节点的本轮输出（追问/新题/反馈），由面试图节点产出
      interview_report: 面试结束生成的结构化报告（维度评分/薄弱点/学习建议）
      daily_report: 日报内容
      user_behavior: 用户行为特征（输入频率、消息长度、时间段等），由 GuardAgent 分析
      classification_confidence: 分类置信度 (0.0~1.0)，由 GuardAgent 计算
      classification_feedback: 分类反馈记录，用于持续优化
    """

    user_id: str
    current_message: str
    emotion_label: str
    emotion_score: float
    message_category: str
    secondary_category: str
    retrieved_memories: List[Dict[str, Any]]
    proactive_memories: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    coding_response: str
    career_response: str
    general_response: str
    final_response: str
    interview_mode: bool
    interview_question: str
    interview_score: float
    interview_session: Dict[str, Any]
    interview_response: str
    interview_report: Dict[str, Any]
    daily_report: str
    user_behavior: Dict[str, Any]
    classification_confidence: float
    classification_feedback: List[Dict[str, Any]]
