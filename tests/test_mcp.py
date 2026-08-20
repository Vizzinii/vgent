"""M9 测试：MCP client（fake 传输单测 + 真实 stdio echo server 集成）。

单测注入假 stdio_client/ClientSession，不触网不起进程；
集成测试用 subprocess 起 scripts/mcp_echo_server.py（FastMCP），验证真实协议。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vgent.config import MCPServerConfig
from vgent.mcp.client import (
    MCPError,
    _prefixed,
    call_server_tool,
    list_server_tools,
    load_into_registry,
)
from vgent.tools import ToolRegistry

ECHO_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "mcp_echo_server.py"


# -- fake 传输 ---------------------------------------------------------------


class FakeStdioClient:
    async def __aenter__(self):
        return (object(), object())  # (read, write)

    async def __aexit__(self, *a):
        return False


class FakeSession:
    def __init__(self, tools=None, call_result=None, call_error=None):
        self._tools = tools or []
        self._call_result = call_result
        self._call_error = call_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def initialize(self):
        pass

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name, arguments=None):
        if self._call_error:
            raise self._call_error
        return self._call_result


def _tool(name, description="", input_schema=None):
    return SimpleNamespace(
        name=name,
        title="",
        description=description,
        input_schema=input_schema or {"type": "object", "properties": {}},
    )


def _text_result(text, is_error=False):
    return SimpleNamespace(content=[SimpleNamespace(text=text)], is_error=is_error)


def _patch(monkeypatch, session):
    monkeypatch.setattr("vgent.mcp.client.stdio_client", lambda params: FakeStdioClient())
    monkeypatch.setattr("vgent.mcp.client.ClientSession", lambda read, write: session)


def _cfg(**kw) -> MCPServerConfig:
    return MCPServerConfig(command="python", **kw)


# -- 单测 -------------------------------------------------------------------


def test_prefixed_slug() -> None:
    assert _prefixed("my-server", "readFile") == "my_server_readfile"
    assert _prefixed("echo", "echo") == "echo_echo"


def test_list_server_tools_converts(monkeypatch) -> None:
    _patch(
        monkeypatch,
        FakeSession(tools=[_tool("echo", "Echo 工具", {"type": "object", "properties": {"text": {}}})]),
    )
    tools = list_server_tools(_cfg(), "echo")
    assert len(tools) == 1
    t = tools[0]
    assert t.name == "echo"
    assert t.description == "Echo 工具"
    assert t.parameters == {"type": "object", "properties": {"text": {}}}
    assert t.permission == "exec"  # 默认档位


def test_list_permission_from_config(monkeypatch) -> None:
    _patch(monkeypatch, FakeSession(tools=[_tool("t")]))
    assert list_server_tools(_cfg(permission="read"), "s")[0].permission == "read"


def test_call_success_text(monkeypatch) -> None:
    _patch(monkeypatch, FakeSession(call_result=_text_result("hi there")))
    assert call_server_tool(_cfg(), "echo", "echo", {"text": "hi"}) == "hi there"


def test_call_is_error_marked(monkeypatch) -> None:
    _patch(monkeypatch, FakeSession(call_result=_text_result("boom", is_error=True)))
    out = call_server_tool(_cfg(), "echo", "echo", {})
    assert "执行失败" in out and "boom" in out


def test_call_exception_returns_error_text(monkeypatch) -> None:
    """决策 9：调用异常不抛穿，转错误文本回喂模型。"""
    _patch(monkeypatch, FakeSession(call_error=RuntimeError("broken")))
    out = call_server_tool(_cfg(), "echo", "echo", {})
    assert "调用出错" in out


def test_list_connection_error_raises_mcp_error(monkeypatch) -> None:
    def bad_stdio(params):
        raise RuntimeError("cannot spawn")

    monkeypatch.setattr("vgent.mcp.client.stdio_client", bad_stdio)
    with pytest.raises(MCPError):
        list_server_tools(_cfg(), "echo")


def test_load_into_registry_registers_and_skips_failed(monkeypatch) -> None:
    """好 server 注册（前缀）；坏 server 跳过不阻塞、不注册。"""
    _patch(
        monkeypatch,
        FakeSession(tools=[_tool("do_it", "工具")], call_result=_text_result("ok")),
    )
    reg = ToolRegistry()
    loaded = load_into_registry(reg, {"alpha": _cfg()})
    assert loaded == {"alpha": ["alpha_do_it"]}
    assert reg.schemas()[0]["function"]["name"] == "alpha_do_it"
    assert reg.execute("alpha_do_it", {}) == "ok"

    def bad_stdio(params):
        raise RuntimeError("boom")

    monkeypatch.setattr("vgent.mcp.client.stdio_client", bad_stdio)
    loaded2 = load_into_registry(reg, {"bad": _cfg()})
    assert loaded2 == {"bad": []}
    assert reg.get("bad_do_it") is None


def test_load_two_servers_no_prefix_collision(monkeypatch) -> None:
    _patch(
        monkeypatch,
        FakeSession(tools=[_tool("list")], call_result=_text_result("ok")),
    )
    reg = ToolRegistry()
    loaded = load_into_registry(reg, {"svc1": _cfg(), "svc2": _cfg()})
    assert loaded == {"svc1": ["svc1_list"], "svc2": ["svc2_list"]}
    assert reg.get("svc1_list") is not None and reg.get("svc2_list") is not None


# -- 集成：真实 stdio 协议 -----------------------------------------------------


def test_real_stdio_echo_server() -> None:
    """subprocess 起 FastMCP echo server：list + call 端到端（真实 framing）。"""
    cfg = MCPServerConfig(command=sys.executable, args=[str(ECHO_SCRIPT)])
    tools = list_server_tools(cfg, "echo")
    assert {"echo", "add"} <= {t.name for t in tools}
    assert call_server_tool(cfg, "echo", "echo", {"text": "hello mcp"}) == "hello mcp"
    assert call_server_tool(cfg, "echo", "add", {"a": 2, "b": 3}) == "5"
