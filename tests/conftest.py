"""公共测试桩（全方位测试方案 v2 · A 组结构性前置）。

FakeLLM / ScriptedLLM / _make_ctx / chat 工厂原先在 6+ 个测试文件里重复定义，
现抽到 conftest 供新测试与后续迁移使用（各文件的本地变体在确认签名一致前保留）。
conftest 所在目录由 pytest 自动加入 sys.path（tests 无 __init__.py），
测试文件直接 `from conftest import FakeLLM, ...` 即可。
"""
from __future__ import annotations

from pathlib import Path

from vgent.agent import SessionContext
from vgent.config import PermissionRules
from vgent.context import ContextEngine
from vgent.llm import ChatResult
from vgent.messages import Message, ToolCall, Usage
from vgent.permission import PermissionSystem
from vgent.store import SessionStore
from vgent.tools import ToolRegistry


class FakeLLM:
    """固定回复的假 LLM：记录每次收到的发送列表，不触网。"""

    def __init__(self, reply: str = "你好") -> None:
        self.calls: list[list[Message]] = []
        self.reply = reply

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(list(messages))
        if on_delta:
            on_delta(self.reply)
        return ChatResult(
            messages=[Message("assistant", self.reply)],
            usage=Usage(10, 5, 15),
        )


class ScriptedLLM:
    """按顺序返回预设响应的假 LLM；响应耗尽再调会 IndexError（测试自己保证足量）。"""

    def __init__(self, responses: list[ChatResult]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(list(messages))
        r = self.responses.pop(0)
        if on_delta and r.messages and r.messages[0].content:
            on_delta(r.messages[0].content)
        if on_reasoning and r.messages and r.messages[0].reasoning_content:
            on_reasoning(r.messages[0].reasoning_content)
        return r


def _chat_with_tools(tc: ToolCall, content: str = "") -> ChatResult:
    return ChatResult(
        messages=[Message("assistant", content, tool_calls=[tc])],
        usage=Usage(10, 5, 15),
        tool_calls=[tc],
    )


def _chat_final(content: str = "done") -> ChatResult:
    return ChatResult(messages=[Message("assistant", content)], usage=Usage(20, 5, 25))


class StageClient:
    """记忆管线专用桩：stage1 固定抽取输出；stage2 行为可切换（ok/bad/raise）。"""

    def __init__(self, stage2_mode: str = "ok") -> None:
        import json as _json

        self._json = _json
        self.stage2_mode = stage2_mode
        self.chat_count = 0

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.chat_count += 1
        system = messages[0].content if messages else ""
        if "raw_bullets" in system or "抽取" in system:
            return ChatResult(
                messages=[Message("assistant", self._json.dumps({"raw_bullets": ["事实甲"], "rollout_summary": "摘要甲"}))],
                usage=Usage(1, 1, 2),
            )
        if self.stage2_mode == "raise":
            raise RuntimeError("stage2 down")
        if self.stage2_mode == "bad":
            return ChatResult(messages=[Message("assistant", "不是 JSON")], usage=Usage(1, 1, 2))
        return ChatResult(
            messages=[Message("assistant", self._json.dumps({"MEMORY_md": "v1\n新记忆", "memory_summary_md": "v1\n新总览"}))],
            usage=Usage(1, 1, 2),
        )


def _make_ctx(
    tmp_path: Path,
    llm=None,
    tools: ToolRegistry | None = None,
    permissions: PermissionSystem | None = None,
    engine: ContextEngine | None = None,
    rules: PermissionRules | None = None,
    **kw,
) -> tuple[SessionContext, SessionStore]:
    """标准测试上下文：临时 SQLite + 指定桩；返回 (ctx, store) 方便收尾 close。"""
    store = SessionStore(tmp_path / "db.sqlite")
    sid = store.create_session()
    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=llm or FakeLLM(),
        tools=tools or ToolRegistry(),
        permissions=permissions or PermissionSystem(rules=rules),
        engine=engine or ContextEngine(),
        **kw,
    )
    return ctx, store
