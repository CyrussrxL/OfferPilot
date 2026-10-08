"""
OfferPilot MCP Tool Server —— 自建工具服务（stdio 传输）

职责：
  基于 MCP 官方 SDK（mcp.server.fastmcp.FastMCP）将本地工具暴露为
  标准化 MCP 工具服务，供 Agent 通过 stdio 协议真实调用。

暴露的工具：
  1. execute_code:            代码沙箱（子进程隔离 + 30s 超时）
  2. evaluate_resume:         简历评分（返回 score / strengths / improvements）
  3. get_interview_questions: 面试题库（按主题抽样）

设计理由：
  - 与"HTTP 占位 + 本地降级"的旧方案不同，本 Server 是真实可运行的
    MCP 服务：Agent 侧通过 stdio_client 子进程拉起，走完整 MCP 协议
    （initialize 握手 → list_tools → call_tool）。
  - 工具实现复用 companion_ai.tools 中的本地版本，保证 MCP 版与
    降级版行为一致。
  - 作为独立进程运行，天然实现执行隔离（沙箱崩了不影响主服务）。

启动方式：
  python -m companion_ai.tools.mcp_server        （stdio 模式）
"""

import sys
from pathlib import Path
from typing import Any, Dict

# 确保以 `python -m` 从任意 cwd 启动时都能导入 companion_ai 包
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP

# ---- 复用本地核心实现（只用 _core，避免 server -> tool -> server 循环递归） ----
from companion_ai.tools.career_tools import (
    _evaluate_resume_core,
    _get_interview_questions_core,
    _get_job_requirements_core,
)
from companion_ai.tools.python_executor import execute_python_code

mcp = FastMCP("offerpilot-tools")


@mcp.tool()
def execute_code(code: str, language: str = "python") -> Dict[str, Any]:
    """在安全沙箱中执行代码（子进程隔离，30 秒超时）。

    Args:
        code: 要执行的代码字符串
        language: 编程语言（当前支持 python）

    Returns:
        包含 stdout / stderr / exit_code / execution_time 的字典
    """
    import time

    start = time.time()
    result = execute_python_code.invoke({"code": code})
    result["execution_time"] = round(time.time() - start, 3)
    result["language"] = language
    return result


@mcp.tool()
def evaluate_resume(resume_text: str) -> Dict[str, Any]:
    """对简历文本进行 ATS 风格评分（0-100），返回优点与改进建议。

    Args:
        resume_text: 简历文本内容

    Returns:
        包含 score / score_level / strengths / improvements 的字典
    """
    return _evaluate_resume_core(resume_text)


@mcp.tool()
def get_interview_questions(topic: str = "AI", count: int = 3) -> Dict[str, Any]:
    """按主题获取面试题（AI/算法/编程/系统设计/行为面试）。

    Args:
        topic: 面试主题
        count: 问题数量

    Returns:
        包含 questions 列表的字典
    """
    return _get_interview_questions_core(topic, count)


@mcp.tool()
def get_job_requirements(position: str) -> Dict[str, Any]:
    """查询目标岗位的技能要求（技能 Gap 分析数据源）。

    Args:
        position: 目标岗位名称或方向（如"AI Agent 开发"、"后端"、"大模型算法"）

    Returns:
        包含 core_skills / bonus_skills / focus 的岗位要求字典；
        未匹配时返回 available_positions 列表
    """
    return _get_job_requirements_core(position)


def main() -> None:
    """以 stdio 模式启动 MCP Server。"""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
