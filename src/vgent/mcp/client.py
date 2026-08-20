"""v2 演进 · MCP 集成（M9）：stdio 连本地 MCP server，工具注册进 ToolRegistry。

- 传输：官方 mcp SDK，stdio（本地 server 主流形态；streamable HTTP 留后续）；
- sync 适配：vgent 全 sync（决策 10）——每次调用 asyncio.run 建连→调用→断开
  （本地进程开销可接受，常驻连接留 v2）；
- 工具：list_tools → ToolSchema，名字前缀 `<server>_<tool>` 防冲突；
  permission 默认 exec（外部能力保守需确认，可 per-server 配置）；
- 失败：连接/协议/调用异常不抛穿，转错误文本回喂模型自纠正（决策 9）。
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, Tool

from vgent.config import MCPServerConfig
from vgent.tools import ToolRegistry, ToolSchema

# 工具输出上限（复用内置工具口径，防病态输出撑爆上下文）
_OUTPUT_CAP = 10_000
# 单次连接/调用超时（秒）：本地 server 卡死不能挂住 agent
_CALL_TIMEOUT = 60.0


class MCPError(Exception):
    """MCP 服务器连接/协议错误（工具级失败，转文本回喂模型）。"""


def _slug(name: str) -> str:
    """服务器/工具名 → 安全前缀（小写，非 [a-z0-9_] 替换为 _）。"""
    return re.sub(r"[^a-z0-9_]", "_", name.lower())


def _prefixed(server: str, tool: str) -> str:
    return f"{_slug(server)}_{_slug(tool)}"


@dataclass
class MCPServerTool:
    server: str
    name: str  # 原始工具名（不带前缀）
    description: str
    parameters: dict  # input_schema（JSON Schema）
    permission: str


async def _connect(config: MCPServerConfig):
    """async 连接：spawn 子进程 + 协议握手，yield 已 initialize 的 session。

    errlog 用 SDK 默认（server 的 stderr 会透传到客户端 stderr——server 侧真实错误
    应当可见；server 自己的 INFO 请求日志应由 server 端压低，如 echo server 的
    log_level=WARNING）。
    """
    params = StdioServerParameters(
        command=config.command,
        args=list(config.args),
        cwd=config.cwd,
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


def _run_loop(coro: Callable):
    """在专用事件循环里跑协程；吞掉 mcp 传输在 loop 关闭时的清理噪音。"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(_teardown_silencer)
    try:
        result = loop.run_until_complete(coro())
        # 退出窗口：让 mcp 传输的生成器 athrow/后台任务完成清理
        # （server 进程已在 shutdown 中杀死，I/O 会解阻，几 tick 足够）
        for _ in range(3):
            loop.run_until_complete(asyncio.sleep(0))
        return result
    finally:
        loop.close()


def _teardown_silencer(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """mcp stdio 传输的后台任务在 loop 关闭时会抛
    'Attempted to exit cancel scope in a different task'——asyncio 默认 handler 会把它
    打印成 traceback 噪音（Windows + asyncio + anyio 已知 teardown 问题，工具结果
    不受影响）。只吞这一种 RuntimeError，其余照常交给默认 handler。
    """
    exc = context.get("exception")
    if isinstance(exc, RuntimeError) and "cancel scope" in str(exc):
        return
    loop.default_exception_handler(context)


def _run_timeout(coro: Callable, server: str, what: str):
    """sync 包装：专用 loop 跑协程 + wait_for；异常统一转 MCPError（连接/协议级）。"""

    async def _guarded():
        return await asyncio.wait_for(coro(), _CALL_TIMEOUT)

    try:
        return _run_loop(_guarded)
    except TimeoutError:
        raise MCPError(f"MCP server {server!r} {what} 超时（>{_CALL_TIMEOUT}s）") from None
    except MCPError:
        raise
    except Exception as exc:
        raise MCPError(f"MCP server {server!r} {what} 失败：{exc}") from exc


def list_server_tools(config: MCPServerConfig, server: str) -> list[MCPServerTool]:
    """列出某 server 的工具（async 内部建连，失败抛 MCPError）。"""

    async def _list() -> list[Tool]:
        async for session in _connect(config):
            result = await session.list_tools()
            return list(result.tools)

    tools = _run_timeout(_list, server, "工具列表")
    return [
        MCPServerTool(
            server=server,
            name=t.name,
            description=(t.description or t.title or ""),
            # mcp 1.x 用 inputSchema（驼峰），2.x 用 input_schema——都兼容
            parameters=dict(
                getattr(t, "input_schema", None)
                or getattr(t, "inputSchema", None)
                or {"type": "object"}
            ),
            permission=config.permission,
        )
        for t in tools
    ]


def call_server_tool(
    config: MCPServerConfig, server: str, tool: str, arguments: dict
) -> str:
    """调用远端工具，返回文本结果；失败返回错误文本（决策 9：不抛穿，回喂模型）。"""
    display = _prefixed(server, tool)

    async def _call() -> CallToolResult:
        async for session in _connect(config):
            return await session.call_tool(tool, arguments)

    try:
        result = _run_timeout(_call, server, f"工具 {tool!r} 调用")
    except MCPError as exc:
        return f"MCP 工具 {display} 调用出错：{exc}"
    text = _result_text(result)
    # mcp 1.x 字段是 isError（驼峰），2.x 是 is_error——都兼容
    if getattr(result, "is_error", None) or getattr(result, "isError", False):
        return f"MCP 工具 {display} 执行失败：{text or '（无错误信息）'}"
    return text or f"（MCP 工具 {display} 无文本输出）"


def _result_text(result: CallToolResult) -> str:
    """CallToolResult.content 的文本块拼接（非文本块给占位标记）。"""
    parts: list[str] = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
        else:
            parts.append(f"<{getattr(block, 'type', 'unknown')} 内容>")
    text = "\n".join(p for p in parts if p).strip()
    if len(text) <= _OUTPUT_CAP:
        return text
    return text[:_OUTPUT_CAP] + f"\n...(输出已截断，共 {len(text)} 字符)"


def _make_handler(
    config: MCPServerConfig, server: str, tool: str
) -> Callable[[dict], str]:
    def handler(args: dict) -> str:
        return call_server_tool(config, server, tool, args)

    return handler


def load_into_registry(
    registry: ToolRegistry, servers: dict[str, MCPServerConfig]
) -> dict[str, list[str]]:
    """把每个 server 的工具注册进 registry（前缀防冲突）；失败跳过该 server 不阻塞。

    返回 {server: [带前缀的工具名]}（cli 展示用）。
    """
    loaded: dict[str, list[str]] = {}
    for server, config in servers.items():
        try:
            tools = list_server_tools(config, server)
        except MCPError:
            loaded[server] = []
            continue
        names: list[str] = []
        for t in tools:
            pname = _prefixed(server, t.name)
            registry.register(
                ToolSchema(
                    name=pname,
                    description=f"[MCP:{server}] {t.name}：{t.description or '（无描述）'}",
                    parameters=t.parameters,
                    permission=t.permission,  # type: ignore[arg-type] — load_config 已校验
                ),
                _make_handler(config, server, t.name),
            )
            names.append(pname)
        loaded[server] = names
    return loaded
