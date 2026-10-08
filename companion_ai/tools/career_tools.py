"""
求职辅导工具模块 —— 简历评分、面试题库

三层结构：
  1. _core 纯函数：本地核心实现（无 MCP、无依赖），供 MCP Server 和降级共用
  2. @tool 包装层：MCP_ENABLED 时优先走自建 MCP Tool Server（stdio 协议），
     失败自动降级 _core 本地实现
  3. MCP Server（mcp_server.py）只 import _core，避免 server -> tool -> server
     的循环递归

设计理由：
  - MCP 版与降级版共享同一实现，行为一致；
  - 降级链路真实可测（关掉 MCP_ENABLED 即是纯本地模式）。
"""

import json
import os
import random
from typing import Any, Dict, List

from langchain_core.tools import tool

from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


def _load_job_requirements() -> Dict[str, Any]:
    """加载岗位技能要求知识库。"""
    data_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "job_requirements.json",
    )
    try:
        with open(data_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载岗位知识库失败: {e}")
        return {}


def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    通过自建 MCP Tool Server（stdio）调用求职工具，返回解析后的 dict。
    失败时抛异常，由调用方降级。
    """
    import json

    from companion_ai.tools.mcp_stdio_client import stdio_mcp_client

    raw = stdio_mcp_client.call_tool(tool_name, arguments, timeout=30)
    data = json.loads(raw) if isinstance(raw, str) else (raw or {})
    if isinstance(data, dict):
        data["source"] = "mcp_stdio_server"
    return data


# ---------------- 本地核心实现（供 MCP Server 与降级共用） ----------------


def _get_job_requirements_core(position: str) -> Dict[str, Any]:
    """岗位技能要求查询核心逻辑：从知识库精确/模糊匹配岗位。

    匹配策略：
      1. 精确匹配岗位名
      2. 模糊匹配（岗位名包含关键词，如"后端"、"算法"）
      3. 返回全部可选岗位列表供 LLM 引导用户明确方向
    """
    library = _load_job_requirements()

    if position in library:
        data = library[position]
        return {
            "success": True,
            "position": position,
            "matched": True,
            **data,
        }

    # 模糊匹配：用户输入是岗位名子串（如"后端"→"后端开发工程师"），
    # 或包含岗位方向关键词（如"做AI应用的"→"大模型应用开发工程师"）
    _DIRECTION_KEYWORDS = {
        "agent": "AI Agent 开发工程师",
        "智能体": "AI Agent 开发工程师",
        "编排": "AI Agent 开发工程师",
        "微调": "大模型算法工程师",
        "训练": "大模型算法工程师",
        "nlp": "NLP 算法工程师",
        "自然语言": "NLP 算法工程师",
        "后端": "后端开发工程师",
        "服务端": "后端开发工程师",
        "应用开发": "大模型应用开发工程师",
        "llm应用": "大模型应用开发工程师",
        "数据": "数据挖掘工程师",
        "测试": "算法测试开发工程师",
        "评测": "算法测试开发工程师",
        "前端": "前端开发工程师",
    }
    position_lower = position.lower()
    for keyword, job_name in _DIRECTION_KEYWORDS.items():
        if keyword in position_lower and job_name in library:
            data = library[job_name]
            return {
                "success": True,
                "position": job_name,
                "matched": True,
                **data,
            }
    for name in library:
        if position and position in name:
            data = library[name]
            return {
                "success": True,
                "position": name,
                "matched": True,
                **data,
            }

    return {
        "success": True,
        "position": position,
        "matched": False,
        "available_positions": list(library.keys()),
        "message": f"知识库中暂无「{position}」的岗位数据，请从 available_positions 中选择",
    }


def _evaluate_resume_core(resume_text: str) -> Dict[str, Any]:
    """简历评分核心逻辑：关键词覆盖 + 结构完整性检查，输出 0-100 分。"""
    try:
        score = 0
        strengths = []
        improvements = []

        keywords = {
            "技能": ["Python", "Java", "C++", "算法", "数据结构", "机器学习", "深度学习", "SQL"],
            "项目": ["项目", "实习", "开发", "设计", "实现"],
            "成果": ["优化", "提升", "减少", "增加", "改进", "完成"],
            "经历": ["实习", "工作", "实践", "比赛", "竞赛"],
        }

        keyword_count = 0
        for category, words in keywords.items():
            for word in words:
                if word in resume_text:
                    keyword_count += 1
                    strengths.append(f"包含{category}相关关键词：{word}")

        score = min(100, 40 + keyword_count * 3)

        if len(resume_text) < 200:
            score -= 15
            improvements.append("简历内容偏短，建议补充更多项目细节和成果描述")
        elif len(resume_text) > 500:
            score += 5
            strengths.append("简历内容较为丰富")

        if not any(kw in resume_text for kw in ["github", "GitHub", "链接", "link"]):
            improvements.append("建议添加 GitHub 链接或项目链接")

        if not any(kw in resume_text for kw in ["%", "倍", "ms", "秒", "提升", "减少"]):
            improvements.append("建议添加量化成果，如性能提升百分比、效率提升等")

        score = max(0, min(100, score))

        score_level = "待改进"
        if score >= 80:
            score_level = "优秀"
        elif score >= 60:
            score_level = "良好"

        return {
            "success": True,
            "score": score,
            "score_level": score_level,
            "strengths": strengths[:5] if strengths else ["简历结构基本完整"],
            "improvements": improvements[:5] if improvements else ["保持当前内容，可继续丰富细节"],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "score": 0,
        }


def _get_interview_questions_core(topic: str = "AI", count: int = 3) -> Dict[str, Any]:
    """面试题库核心逻辑：按主题从内置题库抽样。"""
    question_bank = {
        "AI": [
            "请解释什么是过拟合，以及如何防止过拟合？",
            "请说明神经网络中激活函数的作用，并举例几种常见的激活函数？",
            "请解释梯度下降算法及其变种（SGD、Adam）的区别？",
            "什么是 RNN 和 LSTM，它们解决了什么问题？",
            "请解释 Transformer 架构中的自注意力机制？",
        ],
        "算法": [
            "请说明快速排序的时间复杂度和空间复杂度？",
            "请解释动态规划的适用场景和核心思想？",
            "请说明红黑树和 AVL 树的区别？",
            "请解释图的 BFS 和 DFS 遍历算法？",
            "请说明哈希冲突的解决方法？",
        ],
        "编程": [
            "请实现一个单例模式？",
            "请说明 Python 中的 GIL（全局解释器锁）？",
            "请解释进程和线程的区别？",
            "请说明什么是死锁以及如何避免死锁？",
            "请解释 Python 中的装饰器？",
        ],
        "系统设计": [
            "请设计一个高并发的秒杀系统？",
            "请设计一个分布式缓存系统？",
            "请设计一个消息队列系统？",
            "请设计一个短链接服务？",
            "请设计一个实时聊天系统？",
        ],
        "行为面试": [
            "请描述一次你解决的最困难的技术问题？",
            "请举例说明你在团队中如何处理冲突？",
            "请描述一次你学习新技术的经历？",
            "请举例说明你如何应对压力？",
            "请描述一次你主动改进工作流程的经历？",
        ],
    }

    topic = topic if topic in question_bank else "AI"
    questions = random.sample(question_bank[topic], min(count, len(question_bank[topic])))

    return {
        "success": True,
        "topic": topic,
        "questions": questions,
    }


# ---------------- @tool 包装层（MCP 优先 + 本地降级） ----------------


@tool
def evaluate_resume(resume_text: str) -> Dict[str, Any]:
    """
    简历评分工具（ATS 风格，0-100 分）。

    优先通过自建 MCP Tool Server 执行，返回评分、优点、改进建议；
    MCP 不可用时自动降级本地实现。

    Args:
        resume_text: 简历文本内容

    Returns:
        包含评分、优点、改进建议的字典
    """
    if settings.MCP_ENABLED:
        try:
            return _call_mcp_tool("evaluate_resume", {"resume_text": resume_text})
        except Exception as e:
            logger.warning(f"MCP 简历评分失败，降级本地执行: {e}")
    result = _evaluate_resume_core(resume_text)
    result["source"] = "local_fallback"
    return result


@tool
def get_interview_questions(topic: str = "AI", count: int = 3) -> Dict[str, Any]:
    """
    面试题库工具。

    优先通过自建 MCP Tool Server 获取面试题；
    MCP 不可用时自动降级本地题库。

    Args:
        topic: 面试主题（AI/算法/编程/系统设计/行为面试）
        count: 问题数量

    Returns:
        包含面试问题列表的字典
    """
    if settings.MCP_ENABLED:
        try:
            return _call_mcp_tool(
                "get_interview_questions", {"topic": topic, "count": count}
            )
        except Exception as e:
            logger.warning(f"MCP 面试题库调用失败，降级本地执行: {e}")
    result = _get_interview_questions_core(topic, count)
    result["source"] = "local_fallback"
    return result


@tool
def get_job_requirements(position: str) -> Dict[str, Any]:
    """
    岗位技能要求查询工具（技能 Gap 分析的数据源）。

    从岗位知识库返回目标岗位的核心技能要求、加分项与考察重点，
    供与用户画像/事实记忆对比生成 Gap 矩阵与学习路径；
    优先通过自建 MCP Tool Server 执行，不可用时降级本地。

    Args:
        position: 目标岗位名称或方向（如"AI Agent 开发"、"后端"、"大模型算法"）

    Returns:
        包含 core_skills / bonus_skills / focus 的岗位要求字典
    """
    if settings.MCP_ENABLED:
        try:
            return _call_mcp_tool("get_job_requirements", {"position": position})
        except Exception as e:
            logger.warning(f"MCP 岗位要求查询失败，降级本地执行: {e}")
    result = _get_job_requirements_core(position)
    result["source"] = "local_fallback"
    return result
