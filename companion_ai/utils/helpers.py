import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def invoke_llm_with_tools(llm, prompt: str, tools: List) -> tuple:
    """
    LLM 工具调用循环（单轮工具调用 + 结果回填）。

    流程：
      1. bind_tools 绑定工具 schema
      2. LLM 决定是否调用工具（Function Calling）
      3. 有 tool_calls → 执行工具（LangChain @tool，内部 MCP 优先 + 本地降级）
         → 结果以 ToolMessage 回填 → LLM 基于工具结果生成最终回复
      4. 无 tool_calls → 直接返回

    Args:
        llm: ChatOpenAI 实例
        prompt: 完整提示词
        tools: LangChain @tool 工具列表

    Returns:
        (最终回复文本, 已执行的工具调用记录列表)
    """
    from langchain_core.messages import HumanMessage, ToolMessage

    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(prompt)

    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return response.content, []

    messages = [HumanMessage(content=prompt), response]
    executed = []
    for tc in tool_calls:
        tool = next((t for t in tools if t.name == tc["name"]), None)
        if tool is None:
            continue
        try:
            result = tool.invoke(tc["args"])
        except Exception as e:  # noqa: BLE001
            result = {"success": False, "error": str(e)}
        executed.append({"tool": tc["name"], "args": tc["args"]})
        messages.append(
            ToolMessage(
                content=safe_json_dumps(result),
                tool_call_id=tc["id"],
            )
        )

    final = llm_with_tools.invoke(messages)
    return final.content, executed


def emotion_intensity(score: float) -> str:
    """
    将情感分数映射为强度等级：
    0~0.4 → 低, 0.4~0.7 → 中, 0.7~1.0 → 高
    """
    if score < 0.4:
        return "低"
    elif score < 0.7:
        return "中"
    else:
        return "高"


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return "{}"


def truncate_text(text: str, max_length: int = 500) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_memories(memories: List[Dict]) -> str:
    """
    将检索到的记忆列表格式化为可读文本，供 LLM 上下文使用。
    事实记忆（fact）标记为 [用户事实]，原文记忆（episode）标记为 [历史对话]。
    """
    if not memories:
        return "暂无相关历史记忆。"
    parts = []
    fact_idx = 0
    ep_idx = 0
    for mem in memories:
        text = mem.get("text", "")
        emotion = mem.get("emotion", "unknown")
        ts = mem.get("timestamp", "")
        if mem.get("memory_type") == "fact":
            fact_idx += 1
            parts.append(f"[用户事实{fact_idx}] {text}")
        else:
            ep_idx += 1
            parts.append(f"[历史对话{ep_idx}] ({ts}, 情绪:{emotion}) {text}")
    return "\n".join(parts)


def format_user_profile(profile: Dict[str, Any]) -> str:
    """
    将用户画像格式化为可读文本。
    """
    if not profile:
        return "暂无用户画像信息。"
    lines = []
    for key, value in profile.items():
        label_map = {
            "learning_goal": "学习目标",
            "current_skill_level": "当前技能水平",
            "job_target": "求职目标",
            "emotional_trend": "近期情绪趋势",
        }
        label = label_map.get(key, key)
        if key == "emotional_trend" and isinstance(value, list):
            value = ", ".join([f"{v:.2f}" for v in value])
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)
