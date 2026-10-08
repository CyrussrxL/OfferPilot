"""
MCP stdio 客户端 —— Agent 侧的真实 MCP 协议接入

职责：
  以子进程方式拉起自建 MCP Tool Server（stdio 传输），维持持久会话，
  向上层（LangChain @tool 包装）提供同步的 call_tool / list_tools 接口。

实现要点：
  - MCP 的 ClientSession 是异步上下文管理器，而 LangGraph 节点是同步的；
    因此在后台线程跑一个专属 asyncio 事件循环，会话在整个循环生命周期内
    保持活跃，跨线程用 run_coroutine_threadsafe 提交调用。
  - 单例模式：进程内只拉起一次 server 子进程，避免重复 spawn。
  - 任何失败（拉起超时/调用异常）都向上抛出，由调用方决定降级。

协议链路（区别于旧的 HTTP 占位方案）：
  Agent @tool → StdioMCPClient.call_tool → run_coroutine_threadsafe
  → ClientSession.call_tool → stdio 管道 → FastMCP Server（子进程）
  → 工具实现 → 结果原路返回
"""

import asyncio
import shlex
import threading
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


class StdioMCPClient:
    """基于 stdio 传输的持久 MCP 客户端（线程安全）。"""

    def __init__(self, server_cmd: Optional[str] = None):
        self._server_cmd = server_cmd or settings.MCP_STDIO_SERVER_CMD
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._start_error: Optional[str] = None

    # ---- 生命周期 ----

    def start(self, timeout: float = 30.0) -> bool:
        """拉起 server 子进程并完成 initialize 握手。"""
        if self._session is not None:
            return True
        if self._thread is not None and self._thread.is_alive():
            # 已在启动中，等待就绪
            return self._ready.wait(timeout=timeout)

        # 全新启动（或 server 崩溃后重启）：
        # 必须重置同步状态——_ready 残留 set 会让握手未完成即返回成功，
        # _stop 是一次性 Event，不重置会让新会话立即退出
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._start_error = None

        self._thread = threading.Thread(
            target=self._run_loop, name="mcp-stdio-client", daemon=True
        )
        self._thread.start()

        if not self._ready.wait(timeout=timeout):
            self._start_error = f"MCP server 启动超时（{timeout}s）: {self._server_cmd}"
            logger.warning(self._start_error)
            return False
        if self._start_error:
            logger.warning(f"MCP server 启动失败: {self._start_error}")
            return False

        logger.info(f"MCP stdio 客户端就绪: {self._server_cmd}")
        return True

    def _run_loop(self) -> None:
        """后台线程：运行专属事件循环并维持 MCP 会话存活。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._session_lifecycle())
        except Exception as e:  # noqa: BLE001
            self._start_error = str(e)
            logger.warning(f"MCP 会话异常退出: {e}")
        finally:
            # 异常路径下清理会话引用，保证下次 start() 走全新启动
            self._session = None
            self._ready.set()

    async def _session_lifecycle(self) -> None:
        """建立 stdio 连接 + initialize 握手，然后保持会话直到 stop。"""
        parts = shlex.split(self._server_cmd)
        params = StdioServerParameters(command=parts[0], args=parts[1:])

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                # 保持上下文存活，直到外部请求停止
                while not self._stop.is_set():
                    await asyncio.sleep(0.2)
                self._session = None

    def stop(self) -> None:
        """请求关闭（守护线程，进程退出时也会自动回收）。"""
        self._stop.set()

    # ---- 调用接口 ----

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出 server 暴露的全部工具（走 MCP list_tools 协议）。"""
        self._ensure_session()

        async def _list():
            result = await self._session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": (t.description or "")[:200],
                }
                for t in result.tools
            ]

        future = asyncio.run_coroutine_threadsafe(_list(), self._loop)
        return future.result(timeout=15)

    def call_tool(self, name: str, arguments: Dict[str, Any], timeout: float = 60.0) -> Any:
        """
        调用 MCP 工具（走 MCP call_tool 协议）。

        Args:
            name: 工具名
            arguments: 工具参数
            timeout: 调用超时（秒）

        Returns:
            工具返回的内容（FastMCP 序列化后的文本）

        Raises:
            RuntimeError: 会话未就绪或调用失败
        """
        self._ensure_session()

        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments), self._loop
        )
        try:
            result = future.result(timeout=timeout)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"MCP call_tool 失败: {name} - {e}") from e

        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP 工具执行出错: {name}")

        # FastMCP 的结构化结果在 content[0].text 中（JSON 字符串）
        if result.content:
            return result.content[0].text
        return None

    def _ensure_session(self) -> None:
        if self._session is None:
            if not self.start():
                raise RuntimeError(
                    f"MCP server 不可用: {self._start_error or '未启动'}"
                )


# 进程内单例
stdio_mcp_client = StdioMCPClient()
