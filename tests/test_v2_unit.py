"""全方位测试方案 v2 · B 组：单元缺口（按方案表格逐条，覆盖现有测试未触的路径）。

模块速查：store / agent / llm / context / tools / permission / snapshot /
pipeline / web / cli。与方案不符的现状在注释中说明（真 bug 用 xfail 标记，
已知行为用断言锁定 + 注释指向评审记录）。
"""
from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
from conftest import FakeLLM, StageClient, _chat_final, _chat_with_tools, _make_ctx
from rich.console import Console

import vgent.context as context_mod
import vgent.web.server as server_mod
from vgent.agent import _note_snapshot_before_write, run_turn
from vgent.config import Config
from vgent.context import ContextEngine
from vgent.llm import ChatResult, LLMClient
from vgent.memory.episodic import EpisodicMemory
from vgent.memory.pipeline import MemoryPipeline
from vgent.memory.store import MemoryFileStore
from vgent.messages import Message, ToolCall, Usage
from vgent.permission import (
    Approval,
    ConfirmResult,
    PermissionSystem,
    _outside_workspace,
    persist_allow,
)
from vgent.snapshot import SnapshotStore
from vgent.store import SessionStore
from vgent.task import TaskPlan, TaskStep
from vgent.tools import ToolSchema, default_tools
from vgent.web.server import HubManager, SessionHub, make_server, run_command

# ===========================================================================
# B-store
# ===========================================================================


def test_get_compact_corrupted_retained_skips(tmp_path, monkeypatch) -> None:
    """B-store：手改 DB 写坏 retained JSON → 恢复会话续聊应跳过压缩记录而非崩溃（V1 修复）。"""
    monkeypatch.chdir(tmp_path)
    ctx, store = _make_ctx(tmp_path)
    run_turn("第一轮", ctx)
    store.upsert_compact(ctx.session_id, "摘要", [Message("user", "u")], 2)
    # 手改 DB：把 retained 写成坏 JSON
    with store._lock:
        store._conn.execute(
            "UPDATE session_compacts SET retained = 'not-json' WHERE session_id = ?",
            (ctx.session_id,),
        )
        store._conn.commit()
    # 期望：跳过坏记录、以全量历史续聊，不抛
    run_turn("第二轮", ctx)
    assert store.get_history(ctx.session_id)[-1].role == "assistant"
    store.close()


def test_messages_index_used_in_query_plan(tmp_path) -> None:
    """B-store：EXPLAIN QUERY PLAN 确认 idx_messages_session 真被使用（F16）。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    store.add_message(sid, Message("user", "hi"))
    with store._lock:
        plan = store._conn.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM messages WHERE session_id = ?", (sid,)
        ).fetchall()
    detail = " ".join(str(row) for row in plan)
    assert "idx_messages_session" in detail, detail
    store.close()


def test_add_messages_partial_failure_atomic(tmp_path) -> None:
    """B-store：add_messages 中途抛错 → 整体回滚，无半批残留（V3 修复）。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    bad = SimpleNamespace()  # 无 role/tool_calls 属性 → add_message 内 AttributeError
    with pytest.raises(AttributeError):
        store.add_messages(sid, [Message("user", "good"), bad, Message("user", "after")])
    assert store.get_history(sid) == []  # 期望：无半批残留
    store.close()


# ===========================================================================
# B-agent
# ===========================================================================


def test_send_with_anchors_order(tmp_path) -> None:
    """B-agent：锚点相对顺序——memory_notes → reflection → plan_nudge → hint →
    instructions（用户前、项目后）→ cwd → memory_summary → 正史。"""
    from vgent.agent import _send_with_anchors

    msgs = [Message("user", "hi")]
    send = _send_with_anchors(
        msgs,
        cwd_anchor=Message("system", "CWD"),
        instruction_anchors=[Message("system", "USER"), Message("system", "PROJECT")],
        hint=Message("system", "HINT"),
        plan_nudge=Message("system", "NUDGE"),
        reflection_note=Message("system", "REFLECT"),
        memory_notes=[Message("system", "MEM-NOTE")],
        memory_summary=Message("system", "SUMMARY"),
    )
    texts = [m.content for m in send]
    assert texts == ["MEM-NOTE", "REFLECT", "NUDGE", "HINT", "USER", "PROJECT", "CWD", "SUMMARY", "hi"]


def test_finalize_plan_llm_exception_silent(tmp_path) -> None:
    """B-agent：_finalize_plan 的收尾 LLM 调用抛异常 → 静默返回，回合正常完成。"""
    ctx, store = _make_ctx(tmp_path)

    class BoomLLM:
        calls = 0

        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            type(self).calls += 1
            if type(self).calls == 1:
                # 首轮：输出计划块（pending）
                return ChatResult(
                    messages=[Message("assistant", '[vgent-plan]\n{"steps": [{"description": "s1", "status": "pending"}]}\n[/vgent-plan]')],
                    usage=Usage(10, 5, 15),
                )
            if type(self).calls == 2:
                # 首轮收尾 _finalize_plan：正常纯文本
                return ChatResult(messages=[Message("assistant", "ok")], usage=Usage(10, 5, 15))
            if type(self).calls == 3:
                # 第二轮主调用：纯文本（无计划块）→ 触发收尾 _finalize_plan
                return ChatResult(messages=[Message("assistant", "做完了")], usage=Usage(10, 5, 15))
            raise RuntimeError("llm down")  # 第二轮收尾调用炸掉

    ctx.llm = BoomLLM()
    run_turn("任务", ctx)
    run_turn("继续", ctx)  # 第二轮触发 _finalize_plan → 异常被吞
    assert ctx.state.value == "completed"
    store.close()


def test_reflection_note_injected_not_persisted(tmp_path, monkeypatch) -> None:
    """B-agent：失败工具 → 反思 note 注入下一轮 send（[反思] system）且不落库。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    ctx, store = _make_ctx(tmp_path)
    ctx.tools = default_tools()

    class ReflectLLM:
        def __init__(self):
            self.calls: list[list[Message]] = []

        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            self.calls.append(list(messages))
            if len(self.calls) == 1:  # 首轮：读不存在文件（失败 → 反思）
                return _chat_with_tools(ToolCall("c1", "read_file", '{"path": "nope.txt"}'))
            if len(self.calls) == 2:  # 反思调用
                return ChatResult(messages=[Message("assistant", "Failure: 文件不存在\nAction: 用 search 找")], usage=Usage(5, 2, 7))
            return _chat_final("好了")

    llm = ReflectLLM()
    ctx.llm = llm
    run_turn("读文件", ctx)
    # 第三次 chat（工具失败后的续轮）的 send 含 [反思]，且反思内容不在 SQLite
    third = llm.calls[2]
    assert any(m.role == "system" and m.content.startswith("[反思]") and "Action:" in m.content for m in third)
    persisted = store.get_history(ctx.session_id)
    assert not any(m.role == "system" and m.content.startswith("[反思]") for m in persisted)
    store.close()


def test_auto_memory_title_fallback(tmp_path, monkeypatch) -> None:
    """B-agent：_maybe_auto_memory 标题回退——store 无 title 时用 steps[0].description。

    注：`if plan.steps else "会话"` 分支不可达（plan.done 要求 steps 非空），
    实际可达回退 = get_title() 为空 → steps[0].description（顺带发现该死分支，见报告）。
    """
    monkeypatch.chdir(tmp_path)
    ctx, store = _make_ctx(tmp_path)
    mem = EpisodicMemory(tmp_path / "mem.jsonl")
    ctx.memory = mem
    ctx.memory_auto = True
    ctx.plan = TaskPlan(steps=[TaskStep("步骤甲", "done")])  # 全 done 才触发自动存
    store.update_title(ctx.session_id, "")  # 构造 get_title() → ""

    captured = {}

    def fake_summarize(msgs, llm, topic):
        captured["topic"] = topic
        return "x" * 30  # 过最短长度

    import vgent.agent as agent_mod

    monkeypatch.setattr(agent_mod, "summarize", fake_summarize)
    agent_mod._maybe_auto_memory(ctx, [Message("user", "u"), Message("assistant", "a")])
    assert captured["topic"] == "步骤甲"
    assert mem.count() == 1
    store.close()


class TestNoteSnapshotBeforeWrite:
    def test_gitbash_style_path_skipped(self, tmp_path, monkeypatch) -> None:
        """B-agent：/c/... Git Bash 风格绝对路径在 Windows 上解析越界 → 不登记不崩。"""
        monkeypatch.chdir(tmp_path)
        snaps = SnapshotStore(tmp_path / "ck", tmp_path)
        ctx, store = _make_ctx(tmp_path)
        ctx.snapshots = snaps
        _note_snapshot_before_write(ctx, "write_file", {"path": "/c/Users/x/evil.txt"})
        assert snaps._open_files == {}  # 越界跳过
        store.close()

    def test_abs_path_inside_cwd_registered(self, tmp_path, monkeypatch) -> None:
        """B-agent：cwd 内绝对路径正常登记。"""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "f.txt").write_text("old", encoding="utf-8")
        snaps = SnapshotStore(tmp_path / "ck", tmp_path)
        ctx, store = _make_ctx(tmp_path)
        ctx.snapshots = snaps
        _note_snapshot_before_write(ctx, "write_file", {"path": str(tmp_path / "f.txt")})
        assert "f.txt" in snaps._open_files
        store.close()

    def test_relative_path_registered(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "g.txt").write_text("old", encoding="utf-8")
        snaps = SnapshotStore(tmp_path / "ck", tmp_path)
        ctx, store = _make_ctx(tmp_path)
        ctx.snapshots = snaps
        _note_snapshot_before_write(ctx, "edit_file", {"path": "g.txt"})
        assert "g.txt" in snaps._open_files
        store.close()


# ===========================================================================
# B-llm
# ===========================================================================


def _chunk(delta=None, usage=None):
    if usage is not None:
        return SimpleNamespace(usage=usage, choices=[])
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta)])


def _tool_chunk(id_, name, args, index=0):
    return SimpleNamespace(index=index, id=id_, function=SimpleNamespace(name=name, arguments=args))


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.kwargs = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self._chunks)


def _client(chunks):
    llm = LLMClient(Config())
    fake = _FakeCompletions(chunks)
    llm._client.chat.completions.create = fake.create
    return llm, fake


def test_llm_phantom_slot_merged_when_id_arrives_late() -> None:
    """B-llm：无 id 续片先到、id 后到 → 占位槽（idx:N phantom）并入 id 槽。"""
    llm, _ = _client(
        [
            _chunk(delta=SimpleNamespace(content=None, reasoning_content=None, tool_calls=[_tool_chunk(None, "sh", '{"a')])),
            _chunk(delta=SimpleNamespace(content=None, reasoning_content=None, tool_calls=[_tool_chunk("call_1", None, '": 1}')])),
            _chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
        ]
    )
    result = llm.chat([Message("user", "hi")])
    assert len(result.tool_calls) == 1  # 占位槽被并入，不产生第二个工具调用
    tc = result.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "sh"
    assert tc.arguments == '{"a": 1}'


def test_llm_retry_with_none_on_delta_no_crash(monkeypatch) -> None:
    """B-llm：重试发生时 on_delta=None → 不发提示不崩（F13 只在有回调时提示）。"""
    import httpx
    from openai import APIConnectionError

    monkeypatch.setattr(time, "sleep", lambda s: None)
    attempts = {"n": 0}

    def flaky_create(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise APIConnectionError(request=httpx.Request("POST", "http://x"))
        return iter([_chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))])

    llm = LLMClient(Config())
    llm._client.chat.completions.create = flaky_create
    result = llm.chat([Message("user", "hi")], on_delta=None)  # 不应崩
    assert result.usage is not None and attempts["n"] == 2


def test_llm_usage_last_chunk_wins() -> None:
    """B-llm：usage 出现在中间 chunk 会被末块覆盖（以最后收到的为准）。"""
    llm, _ = _client(
        [
            _chunk(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
            _chunk(delta=SimpleNamespace(content="hi", reasoning_content=None, tool_calls=None)),
            _chunk(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)),
        ]
    )
    result = llm.chat([Message("user", "hi")])
    assert result.usage is not None and result.usage.total_tokens == 15


# ===========================================================================
# B-context
# ===========================================================================


def test_tiktoken_unavailable_falls_back(monkeypatch) -> None:
    """B-context：全局不可用标志置位后 estimate 走启发式回退（含 fixed_extra）。"""
    monkeypatch.setattr(context_mod, "_tiktoken_unavailable", True)
    eng = ContextEngine(context_length=1000)
    msgs = [Message("user", "abcdefghij")]  # 启发式：4 + 10//3 = 7
    extra = "0123456789"  # 10 // 3 = 3
    assert eng.estimate_send_tokens(msgs, model=None, fixed_extra=extra) == 7 + 3


def test_should_compress_estimated_consistent_with_should_compress() -> None:
    """B-context：中段文本下两条触发路径结论一致（都压 / 都不压）。"""
    small = [Message("user", "短"), Message("assistant", "答")]
    big = [Message("user", "长" * 6000), Message("assistant", "答" * 6000)]
    for msgs in (small, big):
        eng = ContextEngine(context_length=1000)
        eng._sync_estimate(msgs)
        heuristic = eng.should_compress()
        estimated = eng.should_compress_estimated(msgs, model=None, fixed_extra=None)
        assert heuristic == estimated, f"两条路径结论不一致：{len(msgs)} 条消息"


def test_hard_limit_drops_head_keeps_last() -> None:
    """B-context【已知行为锁定】：极端超窗时硬下限 pop(0) 先丢头部、保最后一条。

    现行为：compress 内 while 超窗从最旧丢起。锁定此行为防止无意识改动
    （评审记录：决策 8 ③ OpenManus 硬下限兜底）。小尾部预算让中间段可压缩、
    压缩后仍超窗触发硬下限。
    """
    from vgent.config import ContextConfig

    eng = ContextEngine(context_length=60, cfg=ContextConfig(tail_token_budget=10))
    msgs = [Message("system", "h0")] + [Message("user", "u" * 500) for _ in range(3)]
    out = eng.compress(msgs, force=True)
    assert out, "至少保留一条"
    assert out[-1] is msgs[-1]  # 最后一条永远保留
    assert msgs[0] not in out  # 头部被丢（pop(0) 从最旧丢起）


# ===========================================================================
# B-tools
# ===========================================================================


def test_edit_file_reverse_crlf_no_match(tmp_path) -> None:
    """B-tools：old 带 \\r\\n 打 LF 文件 → 不做反向换算 → 报错回喂不写盘。"""
    lf = tmp_path / "lf.txt"
    lf.write_bytes(b"alpha\ngamma\n")
    reg = default_tools()
    out = reg.execute(
        "edit_file",
        {"path": str(lf), "old_string": "alpha\r\ngamma", "new_string": "A\r\nG"},
    )
    assert "未找到" in out
    assert lf.read_bytes() == b"alpha\ngamma\n"


def test_write_file_target_is_directory(tmp_path) -> None:
    """B-tools：write_file 目标是已存在目录 → OSError 分支返回错误文本不崩。"""
    reg = default_tools()
    out = reg.execute("write_file", {"path": str(tmp_path), "content": "x"})
    assert out.startswith("写入失败：")


def test_search_unreadable_file_skipped(tmp_path, monkeypatch) -> None:
    """B-tools：search 遇不可读文件（PermissionError）跳过不崩，其余文件仍可命中。"""
    from pathlib import Path

    ok = tmp_path / "ok.txt"
    ok.write_text("needle here", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("needle blocked", encoding="utf-8")
    orig_read_text = Path.read_text

    def fake_read_text(self, *a, **kw):
        if self == bad:
            raise PermissionError("denied")
        return orig_read_text(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    reg = default_tools()
    out = reg.execute("search", {"pattern": "needle", "path": str(tmp_path)})
    assert "ok.txt" in out
    assert "bad.txt" not in out  # 不可读的被跳过


@pytest.mark.parametrize("offset", [0, -5])
def test_read_file_offset_normalized(tmp_path, offset) -> None:
    """B-tools：read_file offset=0/负数归一到 1（从首行读起）。"""
    f = tmp_path / "f.txt"
    f.write_text("l1\nl2\n", encoding="utf-8")
    reg = default_tools()
    out = reg.execute("read_file", {"path": str(f), "offset": offset})
    assert "l1" in out and "l2" in out


# ===========================================================================
# B-permission
# ===========================================================================


class TestOutsideWorkspaceMatrix:
    """B-permission：F8 _outside_workspace 边界矩阵（A2 参数化）。"""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("sub/x.txt", False),                       # cwd 内相对
            (None, False),                              # 无 path
            (5, False),                                 # 非 str（int）不崩
            (["a"], False),                             # 非 str（list）不崩
            ("../evil.txt", True),                      # 越出 cwd
            ("sub/../../evil.txt", True),               # 归一化后越出
            (r"C:\Windows\evil.txt", True),             # 绝对路径 cwd 外
            (r"\\server\share\x", True),                # UNC
            (r"C:\..\evil.txt", True),                  # 盘根逃逸
        ],
    )
    def test_matrix(self, tmp_path, monkeypatch, raw, expected) -> None:
        monkeypatch.chdir(tmp_path)
        assert _outside_workspace({"path": raw}) is expected

    def test_abs_inside_cwd(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert _outside_workspace({"path": str(tmp_path / "x.txt")}) is False

    def test_drive_letter_case_insensitive(self, tmp_path, monkeypatch) -> None:
        """盘符大小写不影响判定（PureWindowsPath 大小写不敏感）。"""
        monkeypatch.chdir(tmp_path)
        p = str(tmp_path / "x.txt")
        flipped = p[0].swapcase() + p[1:]
        assert _outside_workspace({"path": flipped}) is False

    def test_symlink_escape(self, tmp_path, monkeypatch) -> None:
        outside = tmp_path.parent / "outside_target.txt"
        outside.write_text("x", encoding="utf-8")
        link = tmp_path / "link.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("本机无 symlink 权限")
        monkeypatch.chdir(tmp_path)
        assert _outside_workspace({"path": "link.txt"}) is True


def test_persist_allow_crlf_roundtrip(tmp_path) -> None:
    """B-permission：CRLF 换行的 config.toml 经 persist_allow 往返后语义保留。"""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_bytes(
        b'[provider]\r\nactive = "p"\r\n\r\n[permissions]\r\nask = ["read_file"]\r\ndeny = ["shell"]\r\n'
    )
    assert persist_allow(tmp_path, "write_file") is True
    import tomllib

    data = tomllib.loads(cfg_file.read_text(encoding="utf-8"))
    assert data["permissions"]["allow"] == ["write_file"]
    assert data["permissions"]["ask"] == ["read_file"]
    assert data["permissions"]["deny"] == ["shell"]
    assert data["provider"]["active"] == "p"  # 其余段保留


def test_outside_write_confirmed_every_time_even_sticky(tmp_path, monkeypatch) -> None:
    """B-permission：sticky 放行后越界写盘每次仍真问（不吃 sticky 短路）。"""
    monkeypatch.chdir(tmp_path)
    asked: list[str] = []

    def confirm(tool, args):
        asked.append(str(args.get("path")))
        return ConfirmResult.APPROVE

    perms = PermissionSystem(confirm=confirm)
    perms.approve_sticky("write_file")
    schema = ToolSchema("write_file", "w", {}, "write")
    for _ in range(2):
        assert perms.check(schema, {"path": "../evil.txt"}) is Approval.NEED_CONFIRM
        assert perms.confirm(schema, {"path": "../evil.txt"}) is ConfirmResult.APPROVE
    assert len(asked) == 2  # 两次都真问了
    # cwd 内第二次不再问（sticky 生效）
    assert perms.check(schema, {"path": "in.txt"}) is Approval.AUTO
    assert perms.confirm(schema, {"path": "in.txt"}) is ConfirmResult.APPROVE
    assert len(asked) == 2


# ===========================================================================
# B-snapshot
# ===========================================================================


def test_restore_undo_twice_repeats_same_restore(tmp_path) -> None:
    """B-snapshot【已知行为锁定】：连续两次 restore_undo 恢复同一档
    （pre_restore 只有一档、undo 自身不写 pre），不能连续回退两步。"""
    cwd = tmp_path / "ws"
    cwd.mkdir()
    f = cwd / "f.txt"
    f.write_text("v1", encoding="utf-8")
    snaps = SnapshotStore(tmp_path / "ck", cwd)
    snaps.note_before_write("f.txt")
    f.write_text("v2", encoding="utf-8")
    snaps.seal_turn()
    snaps.restore_last()  # v2 → v1（保存 pre=v2）
    assert f.read_text(encoding="utf-8") == "v1"
    f.write_text("v3", encoding="utf-8")
    r2 = snaps.restore_undo()  # 回 pre（v2）
    assert "f.txt" in r2.restored
    assert f.read_text(encoding="utf-8") == "v2"
    snaps.restore_undo()  # 再次 undo：pre 未变 → 重复恢复到 v2
    assert f.read_text(encoding="utf-8") == "v2"


def test_named_slug_collision_overwrites(tmp_path) -> None:
    """B-snapshot【已知行为锁定】：不同名 slug 相同（"x y" vs "x_y"）→ 后写覆盖前档。"""
    cwd = tmp_path / "ws"
    cwd.mkdir()
    f = cwd / "f.txt"
    f.write_text("A", encoding="utf-8")
    snaps = SnapshotStore(tmp_path / "ck", cwd)
    snaps.note_before_write("f.txt")
    snaps.save_named("x y")
    f.write_text("B", encoding="utf-8")
    snaps.save_named("x_y")  # 与 "x y" 同 slug → 同一 json 文件
    named = list((tmp_path / "ck" / "snapshots" / "named").glob("*.json"))
    assert len(named) == 1  # 前档被覆盖（后写胜出）
    snaps.restore_named("x y")  # exact slug 命中同一文件 → 内容是后拍的 B
    assert f.read_text(encoding="utf-8") == "B"


def test_capture_missing_file_marked_not_crash(tmp_path) -> None:
    """B-snapshot：登记后文件被删 → capture/save_named 打 missing 标不崩。"""
    cwd = tmp_path / "ws"
    cwd.mkdir()
    f = cwd / "gone.txt"
    f.write_text("x", encoding="utf-8")
    snaps = SnapshotStore(tmp_path / "ck", cwd)
    snaps.note_before_write("gone.txt")
    f.unlink()
    snaps.save_named("after-delete")  # 不崩
    files = snaps.capture_session_now()
    assert files["gone.txt"].get("missing") is True


# ===========================================================================
# B-pipeline
# ===========================================================================


def _rc(session_id="s1", text="这是一段足够长的用户输入内容用于抽取"):
    from vgent.memory.pipeline import RoundContent

    return RoundContent(workspace="/w", session_id=session_id, user_text=text)


def test_drain_bounded_retries_on_persistent_stage2_failure(tmp_path) -> None:
    """B-pipeline：stage2 持续失败 → drain 有界重试（DRAIN_MAX_ROUNDS）后放弃，不无限循环。"""
    from vgent.memory.pipeline import DRAIN_MAX_ROUNDS

    store = MemoryFileStore(tmp_path, tmp_path)
    client = StageClient(stage2_mode="bad")
    pipe = MemoryPipeline(store, client, "m", consolidate_min_signals=1, consolidate_idle_seconds=9999)
    pipe.submit(_rc())
    pipe.drain()
    assert client.chat_count >= 2 + DRAIN_MAX_ROUNDS  # stage1 + 每 drain 轮一次 stage2 重试
    assert pipe.pending_signal_count >= 1  # 失败保留 pending
    # drain 有限时间内返回（测试本身不超时即通过）


def test_worker_restart_after_failure(tmp_path, monkeypatch) -> None:
    """B-pipeline：worker 处理失败（写盘抛错）退出后，再次 submit 能重拉 worker 处理。

    注意：client.chat 的异常会被 _chat_text 吞掉（返回空串），worker 真正失败
    要靠 store 写盘抛错触发（F14 丢弃路径）。
    """
    store = MemoryFileStore(tmp_path, tmp_path)
    broken = {"v": True}
    orig_append = MemoryFileStore.append_raw

    def maybe_boom(self, body):
        if broken["v"]:
            raise RuntimeError("disk full")
        return orig_append(self, body)

    monkeypatch.setattr(MemoryFileStore, "append_raw", maybe_boom)
    pipe = MemoryPipeline(store, StageClient(), "m", consolidate_min_signals=99, consolidate_idle_seconds=9999)
    pipe.submit(_rc("第一轮长内容甲"))
    deadline = time.time() + 5
    while pipe.is_running and time.time() < deadline:
        time.sleep(0.05)
    assert not pipe.is_running  # worker 失败退出（F14：批次丢弃）
    broken["v"] = False  # 修复磁盘
    pipe.submit(_rc("第二轮长内容乙"))
    pipe.drain()
    raw = store.root / "raw_memories.md"
    assert "第二轮长内容乙" in raw.read_text(encoding="utf-8")


def test_drop_pending_prefix_fallback_remove(tmp_path) -> None:
    """B-pipeline：头部不匹配时逐条 remove 回退（并发追加的新信号保留）。"""
    store = MemoryFileStore(tmp_path, tmp_path)
    pipe = MemoryPipeline(store, StageClient(), "m")
    s1, s2, s3 = "sig-1", "sig-2", "sig-3"
    pipe._pending_signals = [s1, s2, s3]
    pipe._drop_pending_prefix([s2, s3])  # 头部是 s1 ≠ s2 → 走 remove 回退
    assert pipe._pending_signals == [s1]


# ===========================================================================
# B-web
# ===========================================================================


def _hub(tmp_path, llm=None) -> tuple[HubManager, SessionStore, SessionHub, str]:
    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    manager = HubManager(cfg, store, llm or FakeLLM())
    sid = store.create_session()
    return manager, store, manager.hub(sid), sid


def test_run_command_unknown(tmp_path) -> None:
    """B-web：未知命令返回提示文本（不抛、不落库）。"""
    _, store, hub, sid = _hub(tmp_path)
    out = run_command("/nope", hub)
    assert "未知命令" in out
    assert store.get_history(sid) == []
    store.close()


def test_get_session_detail_fields(tmp_path) -> None:
    """B-web：GET /api/sessions/<sid> 响应字段齐全（plan/running/provider）。"""
    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    manager = HubManager(cfg, store, FakeLLM())
    sid = store.create_session()
    httpd = make_server(manager, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/sessions/{sid}", timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        assert r.status == 200
        assert data["session"]["id"] == sid
        assert "state" in data and "messages" in data
        assert "plan" in data and "running" in data
        assert "provider_name" in data and "provider_model" in data
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_stream_404_closes_connection(tmp_path) -> None:
    """B-web：_stream 不存在的会话 → 404 + 关闭连接（close_connection 置位路径）。"""
    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    manager = HubManager(cfg, store, FakeLLM())
    httpd = make_server(manager, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        try:
            urllib.request.urlopen(f"{base}/api/sessions/deadbeef/stream", timeout=10)
            raised = False
        except urllib.error.HTTPError as e:
            raised = True
            assert e.code == 404
            body = json.loads(e.read().decode("utf-8"))
            assert body["error"] == "session not found"
        assert raised
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_sse_idle_heartbeat(tmp_path, monkeypatch) -> None:
    """B-web：SSE 空闲心跳——HEARTBEAT 调小后连接上能读到 `: ping`。"""
    monkeypatch.setattr(server_mod, "HEARTBEAT", 0.3)
    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    manager = HubManager(cfg, store, FakeLLM())
    sid = store.create_session()
    httpd = make_server(manager, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    got_ping = threading.Event()
    try:
        def reader():
            try:
                with urllib.request.urlopen(f"{base}/api/sessions/{sid}/stream", timeout=10) as r:
                    buf = b""
                    while not got_ping.is_set():
                        chunk = r.read(1)
                        if not chunk:
                            break
                        buf += chunk
                        if b": ping" in buf:
                            got_ping.set()
            except OSError:
                pass  # 测试收尾时 server 先关导致的 socket 竞态（心跳断言已在此之前完成）

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        assert got_ping.wait(timeout=5), "5 秒内未收到心跳 ping"
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


# ===========================================================================
# B-cli
# ===========================================================================


def _fake_prompt(responses: list[str]):
    it = iter(responses)

    def prompt(ps: str) -> str:
        return next(it)

    return prompt


class TestPickSession:
    def test_invalid_input_loops_then_valid(self, tmp_path) -> None:
        """B-cli：_pick_session 非法输入（非数字/超范围）循环重试，合法编号恢复。"""
        store = SessionStore(tmp_path / "t.db")
        sid = store.create_session()
        console = Console(file=io.StringIO(), force_terminal=False)
        picked = None

        from vgent.cli import _pick_session

        picked = _pick_session(store, console, _fake_prompt(["abc", "9", "1"]))
        assert picked == sid
        store.close()

    def test_zero_returns_last_session(self, tmp_path) -> None:
        store = SessionStore(tmp_path / "t.db")
        sid = store.create_session()
        from vgent.cli import _pick_session

        console = Console(file=io.StringIO(), force_terminal=False)
        assert _pick_session(store, console, _fake_prompt(["0"]), last_sid=sid) == sid
        # 无 last_sid 时 0 无效 → 继续循环
        store.create_session()  # 保证有会话可列
        assert _pick_session(store, console, _fake_prompt(["0", "1"]), last_sid=None) is not None
        store.close()


def test_restore_confirm_n_cancels(tmp_path, monkeypatch) -> None:
    """B-cli：/restore <编号> 确认输 n → 取消恢复，盘上文件不变。"""
    from vgent.cli import _dispatch_command

    monkeypatch.chdir(tmp_path)
    cwd = tmp_path
    f = cwd / "f.txt"
    f.write_text("v1", encoding="utf-8")
    snaps = SnapshotStore(tmp_path / "ck", cwd)
    ctx, store = _make_ctx(tmp_path)
    ctx.snapshots = snaps
    ctx.data_dir = tmp_path
    from vgent.agent import _note_snapshot_before_write

    _note_snapshot_before_write(ctx, "write_file", {"path": "f.txt"})
    f.write_text("v2", encoding="utf-8")
    ctx.snapshots.seal_turn()
    console = Console(file=io.StringIO(), force_terminal=False)
    ok = _dispatch_command(
        "/restore 1", ctx, console, _fake_prompt(["n"]), tmp_path / "last", {"n": 0}
    )
    assert ok is True
    assert f.read_text(encoding="utf-8") == "v2"  # 未恢复
    assert "已取消" in console.file.getvalue()
    store.close()


def test_allow_without_config_toml(tmp_path) -> None:
    """B-cli：data_dir 无 config.toml → /allow 提示本会话生效（不持久化不崩）。"""
    from vgent.cli import _dispatch_command

    ctx, store = _make_ctx(tmp_path)
    ctx.data_dir = tmp_path  # 目录存在但无 config.toml
    console = Console(file=io.StringIO(), force_terminal=False)
    ok = _dispatch_command(
        "/allow write_file", ctx, console, _fake_prompt([]), tmp_path / "last", {"n": 0}
    )
    assert ok is True
    assert "本会话生效" in console.file.getvalue()
    assert not (tmp_path / "config.toml").exists()  # 不创建残缺配置
    assert "write_file" in ctx.permissions.rules.allow  # 内存态已生效
    store.close()


@pytest.mark.parametrize("argv", [["serve", "--port", "12345"], ["--serve", "--port", "12345"]])
def test_serve_alias_parsed(tmp_path, monkeypatch, argv) -> None:
    """B-cli：`vgent serve` 与 `--serve` 两种写法等价（argv 转换 + serve 被调）。"""
    import vgent.cli as cli_mod

    cfg = Config(data_dir=tmp_path)
    monkeypatch.setattr(cli_mod, "load_config", lambda provider=None: cfg)
    called: dict = {}

    def fake_serve(c, port=8477, open_browser=True):
        called["port"] = port
        return 0

    import vgent.web.server as ws

    monkeypatch.setattr(ws, "serve", fake_serve)
    assert cli_mod.main(argv) == 0
    assert called["port"] == 12345


def test_get_history_corrupted_tool_calls_degrades(tmp_path) -> None:
    """V1 同类加固：messages.tool_calls 坏 JSON → get_history 降级为无 tool_calls，不崩。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    store.add_message(
        sid, Message("assistant", "调用", tool_calls=[ToolCall("c1", "shell", "{}")])
    )
    with store._lock:
        store._conn.execute(
            "UPDATE messages SET tool_calls = '{bad' WHERE session_id = ?", (sid,)
        )
        store._conn.commit()
    history = store.get_history(sid)  # 修复前：json.JSONDecodeError 裸抛
    assert len(history) == 1
    assert history[0].tool_calls is None  # 降级为普通消息
    assert history[0].content == "调用"
    store.close()
