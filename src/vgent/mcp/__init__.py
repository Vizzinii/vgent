"""v2 演进 · MCP 集成（M9）：stdio 连本地 MCP server，工具注册进 ToolRegistry。"""
from vgent.mcp.client import (
    MCPError,
    MCPServerTool,
    call_server_tool,
    list_server_tools,
    load_into_registry,
)

__all__ = [
    "MCPError",
    "MCPServerTool",
    "call_server_tool",
    "list_server_tools",
    "load_into_registry",
]
