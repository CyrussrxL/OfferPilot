"""
代码沙箱工具 —— mcp_tools

职责：
  execute_code_sandbox：Agent 侧的代码执行工具（LangChain @tool），
  优先通过自建 MCP Tool Server（stdio 协议）执行，失败自动降级本地子进程。

链路：
  CodingAgent bind_tools → execute_code_sandbox
    → StdioMCPClient.call_tool（同步桥接，后台线程事件循环）
    → stdio 管道 → FastMCP Server（python -m companion_ai.tools.mcp_server）
    → 子进程沙箱执行 → 结果原路返回（source="mcp_stdio_server"）

降级：
  MCP 不可用时走 python_executor 本地子进程执行（source="local_fallback"），
  行为一致可测（MCP_ENABLED=false 即纯本地模式）。
"""

from typing import Any, Dict

from langchain_core.tools import tool

from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


@tool
def execute_code_sandbox(code: str, language: str = "python") -> Dict[str, Any]:
    """
    在安全沙箱中执行代码。

    优先通过自建 MCP Tool Server（stdio 协议）执行：
    子进程隔离 + 30 秒超时，执行结果（stdout/stderr/耗时）原路返回。
    MCP 服务不可用时自动降级为本地子进程执行，保证功能不中断。

    Args:
        code: 要执行的代码
        language: 编程语言（当前支持 python）

    Returns:
        包含执行结果或错误信息的字典
    """
    # 1) 优先走真实 MCP 协议（自建 Tool Server）
    if settings.MCP_ENABLED:
        try:
            import json

            from companion_ai.tools.mcp_stdio_client import stdio_mcp_client

            raw = stdio_mcp_client.call_tool(
                "execute_code", {"code": code, "language": language}, timeout=60
            )
            data = json.loads(raw) if isinstance(raw, str) else (raw or {})
            logger.info(
                f"MCP 沙箱执行成功: exit_code={data.get('exit_code')}, "
                f"time={data.get('execution_time')}s"
            )
            return {
                "success": data.get("success", data.get("exit_code") == 0),
                "stdout": data.get("stdout", ""),
                "stderr": data.get("stderr", ""),
                "exit_code": data.get("exit_code", 0),
                "execution_time": data.get("execution_time", 0),
                "source": "mcp_stdio_server",
            }
        except Exception as e:
            logger.warning(f"MCP 沙箱调用失败，降级本地执行: {e}")

    # 2) 降级：本地子进程执行
    from companion_ai.tools.python_executor import execute_python_code
    result = execute_python_code.invoke({"code": code})
    if isinstance(result, dict):
        result["source"] = "local_fallback"
    return result
