"""批 2 回归测试（评审修复 F6-F17，见 HANDOFF「修复计划 · 评审修复批 1/2」）。

F6 Web 请求伪造防护（Origin 白名单 + Content-Type）
F7 read_file/edit_file/search 5MB 大小上限
F8 write 档越界路径无视 sticky/allow 强制确认
F9 /plan 参数精确匹配（/plan renew 不再误清计划）
F10 MAX_TOOL_ROUNDS 超限收尾补 _maybe_auto_memory
F11 /remember 无参提示用法
F12 --print 跑完清理一次性会话
F13 流式重试发提示
F14 记忆管线失败批次丢弃不重排
F15 LLM 客户端 120s 超时
F16 messages 表 session 索引
"""
from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import httpx
from openai import APIConnectionError
from rich.console import Console

import vgent.agent as agent_mod
import vgent.cli as cli_mod
import vgent.llm as llm_mod
from vgent.agent import SessionContext, run_turn
from vgent.cli import _dispatch_command
from vgent.config import Config
from vgent.context import ContextEngine
from vgent.llm import ChatResult, LLMClient
from vgent.memory.episodic import EpisodicMemory
from vgent.memory.pipeline import MemoryPipeline, RoundContent
from vgent.memory.store import MemoryFileStore
from vgent.messages import Message, ToolCall, Usage
from vgent.permission import Approval, ConfirmResult, PermissionSystem
from vgent.store import SessionStore
from vgent.tools import ToolSchema, default_tools
from vgent.web.server import HubManager, make_server, run_command


class FakeLLM:
    def __init__(self, reply: str = "回复内容") -> None:
        self.calls: list[list[Message]] = []
        self.reply = reply

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(list(messages))
        return ChatResult(
            messages=[Message("assistant", self.reply)], usage=Usage(10, 5, 15)
        )


def _ctx(tmp_path, llm, **kw) -> tuple[SessionContext, SessionStore]:
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    ctx = SessionContext(
        session_id=sid, store=store, llm=llm, engine=ContextEngine(), **kw
    )
    return ctx, store


# ---------------------------------------------------------------------------
# F6 Web 请求伪造防护
# ---------------------------------------------------------------------------


def _web_server(tmp_path):
    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    manager = HubManager(cfg, store)
    httpd = make_server(manager, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _req(url: str, method: str = "POST", headers: dict | None = None, body: bytes | None = None):
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def test_web_forbidden_origin(tmp_path) -> None:
    """F6：恶意网页的 Origin（非本机）→ POST/DELETE 403。"""
    httpd, base = _web_server(tmp_path)
    try:
        code, _ = _req(
            base + "/api/sessions",
            headers={"Origin": "https://evil.com", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert code == 403
        code, _ = _req(
            base + "/api/sessions/xxx", method="DELETE", headers={"Origin": "https://evil.com"}
        )
        assert code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_web_text_plain_rejected_local_origin_ok(tmp_path) -> None:
    """F6：text/plain 带体 POST（跨域简单请求绕预检的口子）→ 400；
    本机 Origin + JSON、以及无 Origin 的客户端（curl/测试）正常放行。"""
    httpd, base = _web_server(tmp_path)
    try:
        code, _ = _req(
            base + "/api/sessions",
            headers={"Content-Type": "text/plain"},
            body=json.dumps({"content": "hi"}).encode("utf-8"),
        )
        assert code == 400
        code, _ = _req(base + "/api/sessions", body=b"{}")  # 无 Origin、无 Content-Type 但有体
        # urllib 默认 application/x-www-form-urlencoded → 也应 400（非 JSON 声明）
        assert code == 400
        code, _ = _req(
            base + "/api/sessions",
            headers={"Origin": "http://127.0.0.1:9999", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert code == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# F7 读文件大小上限
# ---------------------------------------------------------------------------


def test_read_tools_size_cap(tmp_path) -> None:
    """F7：>5MB 文件 read/edit 拒绝整读；search 单文件目标拒绝、目录搜索跳过。"""
    reg = default_tools()
    big = tmp_path / "big.txt"
    big.write_text("needle" + "x" * (5 * 1024 * 1024 + 1), encoding="utf-8")
    small = tmp_path / "small.txt"
    small.write_text("needle here\n", encoding="utf-8")

    out = reg.execute("read_file", {"path": str(big)})
    assert "5MB" in out and "错误" in out
    out = reg.execute("edit_file", {"path": str(big), "old_string": "y", "new_string": "z"})
    assert "5MB" in out
    assert big.read_text(encoding="utf-8").startswith("needle")  # 未被改动
    out = reg.execute("search", {"pattern": "needle", "path": str(big)})
    assert "5MB" in out
    # 目录搜索：超大文件跳过，小文件照常命中
    out = reg.execute("search", {"pattern": "needle", "path": str(tmp_path)})
    assert "small.txt" in out
    assert "big.txt" not in out


# ---------------------------------------------------------------------------
# F8 write 档越界路径强制确认
# ---------------------------------------------------------------------------

_WRITE = ToolSchema(name="write_file", description="", parameters={}, permission="write")


def test_outside_workspace_write_forces_confirm(tmp_path, monkeypatch) -> None:
    """F8：sticky/allow 放行后，越出 cwd 的写盘目标仍强制确认（confirm 不吃 sticky 短路）。"""
    monkeypatch.chdir(tmp_path)
    ps = PermissionSystem()
    ps.approve_sticky("write_file")
    ps.rules.allow.append("write_file")

    # cwd 内相对路径：sticky/allow 生效，AUTO + confirm 短路
    assert ps.check(_WRITE, {"path": "ok.txt"}) is Approval.AUTO
    assert ps.confirm(_WRITE, {"path": "ok.txt"}) is ConfirmResult.APPROVE

    # 越界绝对路径：无视 sticky/allow → NEED_CONFIRM；无交互 confirm → 拒绝
    outside = tmp_path.parent / "evil.txt"
    assert ps.check(_WRITE, {"path": str(outside)}) is Approval.NEED_CONFIRM
    assert ps.confirm(_WRITE, {"path": str(outside)}) is ConfirmResult.REJECT
    # 相对路径逃逸（..）同理
    assert ps.check(_WRITE, {"path": "../evil.txt"}) is Approval.NEED_CONFIRM
    # read/exec 档不受影响（无 path 概念时 args 无 path → 不越界）
    read_tool = ToolSchema(name="read_file", description="", parameters={}, permission="read")
    assert ps.check(read_tool, {"path": str(outside)}) is Approval.AUTO


# ---------------------------------------------------------------------------
# F9 /plan 参数精确匹配 + F11 /remember 无参用法
# ---------------------------------------------------------------------------


def _plan_text() -> str:
    return (
        "[vgent-plan]\n"
        + json.dumps({"steps": [{"description": "步骤一", "status": "done"}]}, ensure_ascii=False)
        + "\n[/vgent-plan]"
    )


def test_plan_renew_not_redo(tmp_path) -> None:
    """F9：`/plan renew` 是查看（不误清计划）；`/plan new` 才清除。"""
    ctx, store = _ctx(tmp_path, FakeLLM())
    store.upsert_plan_message(ctx.session_id, _plan_text())
    console = Console(file=io.StringIO(), force_terminal=False)
    prompt = lambda ps: ""

    assert _dispatch_command("/plan renew", ctx, console, prompt, tmp_path / "last", {"n": 0})
    assert any("[vgent-plan]" in m.content for m in store.get_history(ctx.session_id))
    assert _dispatch_command("/plan new", ctx, console, prompt, tmp_path / "last", {"n": 0})
    assert not any("[vgent-plan]" in m.content for m in store.get_history(ctx.session_id))
    store.close()


def test_remember_no_args_shows_usage(tmp_path) -> None:
    """F11：`/remember` 无参提示用法（CLI + Web 双端），不再跳去列记忆。"""
    ctx, store = _ctx(tmp_path, FakeLLM(), memory=EpisodicMemory(tmp_path / "mem.jsonl"))
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    assert _dispatch_command("/remember", ctx, console, lambda ps: "", tmp_path / "last", {"n": 0})
    assert "用法：/remember" in buf.getvalue()

    hub = SimpleNamespace(ctx=ctx)
    assert "用法：/remember" in run_command("/remember", hub)
    assert "还没有任何历史记忆" in run_command("/memories", hub)
    store.close()


# ---------------------------------------------------------------------------
# F10 超限收尾补 _maybe_auto_memory
# ---------------------------------------------------------------------------


def test_overlimit_path_calls_auto_memory(tmp_path, monkeypatch) -> None:
    """F10：MAX_TOOL_ROUNDS 溢出收尾也走 _maybe_auto_memory（计划全 done → 自动存摘要）。"""
    monkeypatch.setattr(agent_mod, "MAX_TOOL_ROUNDS", 2)
    tc = ToolCall(id="c1", name="echo", arguments="{}")
    plan_done = _plan_text()
    summary_reply = ChatResult(
        messages=[Message(
            "assistant",
            "<summary>本轮通过 echo 工具完成了占位任务，计划步骤全部标记完成，"
            "会话按预期收尾，没有遗留事项。</summary>",
        )],
        usage=Usage(5, 5, 10),
    )
    responses = [
        ChatResult(
            messages=[Message("assistant", plan_done, tool_calls=[tc])],
            usage=Usage(10, 5, 15),
            tool_calls=[tc],
        ),
        ChatResult(
            messages=[Message("assistant", plan_done, tool_calls=[tc])],
            usage=Usage(10, 5, 15),
            tool_calls=[tc],
        ),
        ChatResult(messages=[Message("assistant", "done")], usage=Usage(10, 5, 15)),
        summary_reply,  # _maybe_auto_memory 的 summarize 调用
    ]

    class ScriptedLLM:
        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            self.calls.append(list(messages))
            return responses.pop(0)

    llm = ScriptedLLM()
    memory = EpisodicMemory(tmp_path / "mem.jsonl")
    ctx, store = _ctx(tmp_path, llm, memory=memory, memory_auto=True)
    ctx.tools.register(
        ToolSchema(name="echo", description="", parameters={}, permission="read"),
        lambda args: "ok",
    )
    run_turn("跑两轮工具", ctx)
    assert ctx.plan is not None and ctx.plan.done
    assert memory.has_session(ctx.session_id)  # 修复前超限路径漏调 → False
    # 复审跟进：summarize 输入必须含超限收尾回复（修复前缺 final 消息）
    assert any(m.role == "assistant" and m.content == "done" for m in llm.calls[3])
    store.close()


# ---------------------------------------------------------------------------
# F12 --print 会话清理
# ---------------------------------------------------------------------------


def test_print_headless_cleans_session(tmp_path, monkeypatch) -> None:
    """F12：headless 一次性会话跑完（含失败路径）即删，不污染会话列表。"""
    ok = ChatResult(messages=[Message("assistant", "hi")], usage=None)
    monkeypatch.setattr(cli_mod, "run_turn", lambda *a, **k: ok)
    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    console = Console(file=io.StringIO(), force_terminal=False)
    assert cli_mod._run_headless(cfg, store, "你好", console) == 0
    assert store.list_sessions() == []

    def boom(*a, **k):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(cli_mod, "run_turn", boom)
    assert cli_mod._run_headless(cfg, store, "你好", console) == 1
    assert store.list_sessions() == []
    store.close()


# ---------------------------------------------------------------------------
# F13 流式重试提示
# ---------------------------------------------------------------------------


def test_retry_notice_emitted(monkeypatch) -> None:
    """F13：可重试错误发生时经 on_delta 发「重试中」提示（已流出的增量之后）。"""
    monkeypatch.setattr(llm_mod, "_RETRY_BACKOFF", (0.0,))
    client = LLMClient.__new__(LLMClient)  # 不构造真实 OpenAI，_chat_once 直接打桩
    client.cfg = Config()
    client.max_retries = 1
    calls: list[int] = []

    def flaky(kwargs, on_delta, on_reasoning):
        calls.append(1)
        if len(calls) == 1:
            if on_delta:
                on_delta("部分输出")
            raise APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
        return ChatResult(messages=[Message("assistant", "ok")], usage=None)

    monkeypatch.setattr(client, "_chat_once", flaky)
    deltas: list[str] = []
    client.chat([Message("user", "hi")], on_delta=deltas.append)
    assert deltas[0] == "部分输出"
    assert any("重试中" in d for d in deltas)


# ---------------------------------------------------------------------------
# F14 管线失败批次丢弃
# ---------------------------------------------------------------------------


def test_pipeline_drops_failed_batch(tmp_path, monkeypatch) -> None:
    """F14：_process_batch 中途失败（append_raw 磁盘错误）→ 批次丢弃不重排
    （重排会重复写 raw/rollout）；queue 空、无 pending 信号。"""
    mstore = MemoryFileStore(tmp_path / "home", tmp_path / "ws")

    class Stage1LLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            return ChatResult(
                messages=[Message(
                    "assistant",
                    '{"raw_bullets": ["项目约定：Redis 7 二级缓存"], "rollout_summary": "本轮摘要"}',
                )],
                usage=None,
            )

    pipe = MemoryPipeline(
        mstore, Stage1LLM(), "m", consolidate_min_signals=99, consolidate_idle_seconds=9999
    )

    def boom(body):
        raise OSError("disk full")

    monkeypatch.setattr(mstore, "append_raw", boom)
    rc = RoundContent(
        workspace=str(tmp_path), session_id="s1",
        user_text="这是一个足够长的用户问题，值得抽取记忆",
        assistant_texts=("回答正文",),
    )
    pipe.submit(rc)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        worker = pipe._worker
        if worker is not None and not worker.is_alive():
            break
        time.sleep(0.05)
    assert pipe._queue.empty()  # 修复前：批次被塞回队列
    assert pipe.pending_signal_count == 0


# ---------------------------------------------------------------------------
# F15 LLM 超时 + F16 messages 索引
# ---------------------------------------------------------------------------


def test_llm_client_timeout(tmp_path) -> None:
    """F15：OpenAI 客户端显式 120s 超时（SDK 默认 600s 太长）。"""
    client = LLMClient(Config(data_dir=tmp_path))
    assert client._client.timeout == 120.0


def test_messages_index_created(tmp_path) -> None:
    """F16：messages(session_id) 索引存在（get_history 不再全表扫描）。"""
    store = SessionStore(tmp_path / "t.db")
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_session'"
    ).fetchone()
    assert row is not None
    store.close()


# ---------------------------------------------------------------------------
# 复审跟进（批 2 严审发现的缺口）
# ---------------------------------------------------------------------------


def test_edit_file_preserves_line_endings(tmp_path) -> None:
    """复审跟进：字节级保真——LF 文件编辑后仍是 LF（不再整文件变 CRLF）；
    CRLF 文件用 \\n 风格 old_string 也能匹配且保持 CRLF。"""
    reg = default_tools()
    lf = tmp_path / "lf.txt"
    lf.write_bytes(b"hello abc\nsecond line\n")
    out = reg.execute("edit_file", {"path": str(lf), "old_string": "abc", "new_string": "xyz"})
    assert "已替换 1 处" in out
    assert lf.read_bytes() == b"hello xyz\nsecond line\n"  # LF 原样

    crlf = tmp_path / "crlf.txt"
    crlf.write_bytes(b"alpha beta\r\ngamma\r\n")
    out = reg.execute(
        "edit_file",
        {"path": str(crlf), "old_string": "alpha beta\ngamma", "new_string": "A\nG"},
    )
    assert "已替换 1 处" in out  # \n 风格 old_string 命中 CRLF 文件
    assert crlf.read_bytes() == b"A\r\nG\r\n"  # CRLF 原样


def test_origin_null_rejected(tmp_path) -> None:
    """复审跟进：Origin: null（sandboxed iframe 可发出）→ 403，不再绕过白名单。"""
    httpd, base = _web_server(tmp_path)
    try:
        code, _ = _req(
            base + "/api/sessions",
            headers={"Origin": "null", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert code == 403
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_host_header_validated(tmp_path) -> None:
    """复审跟进：Host 非本机（DNS rebinding）→ GET/POST 403；默认 Host 正常。"""
    httpd, base = _web_server(tmp_path)
    try:
        code, _ = _req(
            base + "/api/sessions", headers={"Host": "evil.com"}
        )  # GET（urllib 不带 method 默认 GET，无 body）
        assert code == 403
        code, _ = _req(
            base + "/api/sessions",
            method="POST",
            headers={"Host": "evil.com", "Content-Type": "application/json"},
            body=b"{}",
        )
        assert code == 403
        with urllib.request.urlopen(base + "/api/sessions", timeout=10) as r:
            assert r.status == 200  # 正常 Host 放行
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_headless_construction_failure_cleans_session(tmp_path, monkeypatch) -> None:
    """复审跟进：构造段（LLMClient 等）失败 → rc 1 且不留残留会话。"""
    def boom(cfg):
        raise RuntimeError("ctor failed")

    monkeypatch.setattr(cli_mod, "LLMClient", boom)
    store = SessionStore(tmp_path / "t.db")
    console = Console(file=io.StringIO(), force_terminal=False)
    assert cli_mod._run_headless(Config(data_dir=tmp_path), store, "hi", console) == 1
    assert store.list_sessions() == []
    store.close()
