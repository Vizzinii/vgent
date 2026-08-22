"""M12-B 测试：快照/恢复（SnapshotStore 单测 + run_turn/CLI 集成）。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from vgent.agent import SessionContext, run_turn
from vgent.llm import ChatResult
from vgent.messages import Message, ToolCall, Usage
from vgent.permission import ConfirmResult, PermissionSystem
from vgent.snapshot import MAX_NAMED, SnapshotStore
from vgent.store import SessionStore
from vgent.tools import default_tools

# -- 工具 ---------------------------------------------------------------


def _make(tmp_path, cwd_name: str = "ws"):
    cwd = tmp_path / cwd_name
    cwd.mkdir(exist_ok=True)
    store = SnapshotStore(tmp_path / "ckpt" / "sess1", cwd)
    return store, cwd


def _write(cwd: Path, rel: str, content: str) -> None:
    p = cwd / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


# -- SnapshotStore 单测 ---------------------------------------------------


def test_note_before_write_idempotent_same_turn(tmp_path) -> None:
    """同回合同文件只登记一次（幂等）；blob 保存写盘前原文。"""
    store, cwd = _make(tmp_path)
    _write(cwd, "a.txt", "v1")
    store.begin_turn()
    store.note_before_write("a.txt")
    store.note_before_write("a.txt")  # 幂等：第二次不重复
    store.seal_turn()
    assert store.snapshot_count() == 1
    entries = store.list_entries()
    assert entries[0]["files"] == ["a.txt"]
    blobs = list((store.root / "blobs").glob("*"))
    assert len(blobs) == 1
    assert blobs[0].read_bytes() == b"v1"


def test_turn_multiversion_restore_index(tmp_path) -> None:
    """跨回合多版本：restore 回合 N 快照 = 回到该回合写盘前状态。"""
    store, cwd = _make(tmp_path)
    _write(cwd, "a.txt", "init")

    store.begin_turn()  # 回合 1：写 "a1"（登记 init）
    store.note_before_write("a.txt")
    _write(cwd, "a.txt", "a1")
    store.seal_turn()

    store.begin_turn()  # 回合 2：写 "a2"（登记 a1）
    store.note_before_write("a.txt")
    _write(cwd, "a.txt", "a2")
    store.seal_turn()

    assert store.snapshot_count() == 2
    assert (cwd / "a.txt").read_text(encoding="utf-8") == "a2"

    report = store.restore_index(1)  # 最新回合（回合 2 前）→ a1
    assert report.restored == ["a.txt"]
    assert (cwd / "a.txt").read_text(encoding="utf-8") == "a1"

    report = store.restore_index(2)  # 回合 1 前 → init
    assert (cwd / "a.txt").read_text(encoding="utf-8") == "init"


def test_restore_missing_deletes(tmp_path) -> None:
    """登记时文件不存在（missing）→ restore 删当前文件。"""
    store, cwd = _make(tmp_path)
    store.begin_turn()
    store.note_before_write("new.txt")  # 写盘前不存在
    _write(cwd, "new.txt", "created later")
    store.seal_turn()
    report = store.restore_last()
    assert report.deleted == ["new.txt"]
    assert not (cwd / "new.txt").exists()


def test_crash_open_turn_promoted(tmp_path) -> None:
    """崩溃：begin_turn 登记后未 seal → 重建 store 时提升为快照（/restore 可用）。"""
    store, cwd = _make(tmp_path)
    _write(cwd, "a.txt", "v1")
    store.begin_turn()
    store.note_before_write("a.txt")
    # 模拟崩溃：不 seal，直接新建 store（同目录）
    store2 = SnapshotStore(tmp_path / "ckpt" / "sess1", cwd)
    assert store2.snapshot_count() == 1
    _write(cwd, "a.txt", "v2")
    report = store2.restore_last()
    assert report.restored == ["a.txt"]
    assert (cwd / "a.txt").read_text(encoding="utf-8") == "v1"


def test_save_named_and_restore_named(tmp_path) -> None:
    """命名档：拍此刻全量 → 改文件 → restore_named 回档。"""
    store, cwd = _make(tmp_path)
    _write(cwd, "a.txt", "v1")
    store.begin_turn()
    store.note_before_write("a.txt")
    _write(cwd, "a.txt", "v2")
    store.seal_turn()
    name = store.save_named("里程碑")
    assert name == "里程碑"
    _write(cwd, "a.txt", "v3")
    report = store.restore_named("里程碑")
    assert report.restored == ["a.txt"]
    assert (cwd / "a.txt").read_text(encoding="utf-8") == "v2"


def test_named_eviction_keeps_newest(tmp_path) -> None:
    """命名档超上限淘汰最旧（保留最新 MAX_NAMED 个）。"""
    store, cwd = _make(tmp_path)
    _write(cwd, "a.txt", "x")
    store.begin_turn()
    store.note_before_write("a.txt")
    store.seal_turn()
    for i in range(MAX_NAMED + 5):
        store.save_named(f"n{i}")
    named_dir = store.root / "named"
    files = sorted(named_dir.glob("*.json"))
    assert len(files) == MAX_NAMED
    # 最早创建的 n0 应被淘汰
    assert not (named_dir / "n0.json").exists()


def test_restore_undo_roundtrip(tmp_path) -> None:
    """restore 前自动存 pre_restore；/restore undo 回到 restore 之前。"""
    store, cwd = _make(tmp_path)
    _write(cwd, "a.txt", "v1")
    store.begin_turn()
    store.note_before_write("a.txt")
    _write(cwd, "a.txt", "v2")
    store.seal_turn()
    store.save_named("m")
    _write(cwd, "a.txt", "v3")  # restore 前当前状态 = v3

    store.restore_named("m")
    assert (cwd / "a.txt").read_text(encoding="utf-8") == "v2"
    report = store.restore_undo()
    assert report.restored == ["a.txt"]
    assert (cwd / "a.txt").read_text(encoding="utf-8") == "v3"


def test_blob_gc_removes_unreferenced(tmp_path) -> None:
    """无任何记录引用的 blob 被 GC 删除。"""
    store, cwd = _make(tmp_path)
    _write(cwd, "a.txt", "v1")
    store.begin_turn()
    store.note_before_write("a.txt")
    store.seal_turn()
    blob_dir = store.root / "blobs"
    assert len(list(blob_dir.glob("*"))) == 1
    # 清空回合快照（模拟记录被淘汰/手工清理）→ 下回合 seal 触发 GC
    (store.root / "snapshots.json").write_text('{"v":1,"items":[]}', encoding="utf-8")
    store.begin_turn()
    store.seal_turn()
    assert len(list(blob_dir.glob("*"))) == 0


def test_cleanup_old_removes_stale_sessions(tmp_path) -> None:
    """超期未活动的会话快照目录被清理（30 天默认）。"""
    import time

    root = tmp_path / "checkpoints"
    fresh = root / "sess-fresh"
    stale = root / "sess-stale"
    fresh.mkdir(parents=True)
    stale.mkdir(parents=True)
    # 直接把 stale 的 mtime 改旧（31 天前）
    t = time.time() - 31 * 86400
    os.utime(stale, (t, t))
    removed = SnapshotStore.cleanup_old(root, days=30)
    assert removed == 1
    assert not stale.exists()
    assert fresh.exists()


def test_restore_index_out_of_range(tmp_path) -> None:
    store, cwd = _make(tmp_path)
    _write(cwd, "a.txt", "x")
    store.begin_turn()
    store.note_before_write("a.txt")
    store.seal_turn()
    with pytest.raises(ValueError):
        store.restore_index(2)


def test_normalize_name_rejects_reserved_and_separators() -> None:
    from vgent.snapshot import normalize_snapshot_name

    with pytest.raises(ValueError):
        normalize_snapshot_name("last")
    with pytest.raises(ValueError):
        normalize_snapshot_name("a/b")
    assert normalize_snapshot_name("")  # 空名 → 时间戳


# -- run_turn 集成 ---------------------------------------------------------


class ScriptedLLM:
    def __init__(self, responses: list[ChatResult]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(messages)
        return self.responses.pop(0)


def _chat_with_tools(tc: ToolCall) -> ChatResult:
    return ChatResult(
        messages=[Message("assistant", "", tool_calls=[tc])],
        usage=Usage(10, 5, 15),
        tool_calls=[tc],
    )


def _chat_final(content: str = "完成") -> ChatResult:
    return ChatResult(messages=[Message("assistant", content)], usage=Usage(20, 5, 25))


def _agent_ctx(tmp_path, monkeypatch, llm) -> tuple[SessionContext, Path]:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    monkeypatch.chdir(ws)  # 工具与快照的 cwd 基准一致
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=llm,
        tools=default_tools(),
        permissions=PermissionSystem(confirm=lambda t, a: ConfirmResult.APPROVE),
        snapshots=SnapshotStore(tmp_path / "ckpt" / sid, ws),
    )
    return ctx, ws


def test_run_turn_snapshots_note_on_write(tmp_path, monkeypatch) -> None:
    """run_turn 里 write_file 触发登记；回合末封存成快照。"""
    llm = ScriptedLLM(
        [
            _chat_with_tools(ToolCall("c1", "write_file", json.dumps({"path": "a.txt", "content": "hi"}))),
            _chat_final(),
        ]
    )
    ctx, ws = _agent_ctx(tmp_path, monkeypatch, llm)
    run_turn("写个文件", ctx)
    assert (ws / "a.txt").read_text(encoding="utf-8") == "hi"
    assert ctx.snapshots.snapshot_count() == 1
    entries = ctx.snapshots.list_entries()
    assert entries[0]["files"] == ["a.txt"]
    ctx.store.close()


def test_run_turn_snapshots_none_noop(tmp_path, monkeypatch) -> None:
    """snapshots=None 时全路径 no-op（不建目录、不登记）。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    llm = ScriptedLLM(
        [
            _chat_with_tools(ToolCall("c1", "write_file", json.dumps({"path": "a.txt", "content": "hi"}))),
            _chat_final(),
        ]
    )
    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=llm,
        tools=default_tools(),
        permissions=PermissionSystem(confirm=lambda t, a: ConfirmResult.APPROVE),
    )
    run_turn("写个文件", ctx)
    assert (ws / "a.txt").read_text(encoding="utf-8") == "hi"
    assert not (tmp_path / "ckpt").exists()
    store.close()


def test_run_turn_seal_on_exception(tmp_path, monkeypatch) -> None:
    """异常路径也 seal：无残留 open_turn 登记。"""
    llm = ScriptedLLM([])  # 空响应 → IndexError
    ctx, _ws = _agent_ctx(tmp_path, monkeypatch, llm)
    with pytest.raises(IndexError):
        run_turn("会失败", ctx)
    open_data = json.loads((ctx.snapshots.root / "open_turn.json").read_text(encoding="utf-8"))
    assert open_data["files"] == {}
    ctx.store.close()


def test_note_snapshot_skips_outside_cwd(tmp_path) -> None:
    """绝对路径越出 cwd 时不登记（不阻断工具）。"""
    from vgent.agent import _note_snapshot_before_write

    store, _cwd = _make(tmp_path)
    db = SessionStore(tmp_path / "t2.db")
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    ctx = SessionContext(
        session_id="s",
        store=db,
        llm=SimpleNamespace(chat=lambda *a, **k: None),
        snapshots=store,
    )
    _note_snapshot_before_write(ctx, "write_file", {"path": str(outside)})
    assert store.snapshot_count() == 0  # 未登记
    _note_snapshot_before_write(ctx, "read_file", {"path": "a.txt"})  # 非 write/edit：跳过
    assert store.snapshot_count() == 0
    db.close()


# -- CLI 命令 -------------------------------------------------------------


def _cli_ctx(tmp_path) -> tuple[SessionContext, Path]:
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=SimpleNamespace(chat=lambda *a, **k: None),
        data_dir=tmp_path,
    )
    ctx.snapshots = SnapshotStore(tmp_path / "ckpt" / sid, ws)
    return ctx, ws


def test_snapshot_restore_dispatch(tmp_path) -> None:
    """/snapshot 拍档 → 改文件 → /restore 无参列出 + /restore last 恢复。"""
    from vgent.cli import _dispatch_command

    ctx, ws = _cli_ctx(tmp_path)
    _write(ws, "a.txt", "v1")
    ctx.snapshots.begin_turn()
    ctx.snapshots.note_before_write("a.txt")
    _write(ws, "a.txt", "v2")
    ctx.snapshots.seal_turn()
    console = Console()
    last = tmp_path / "last"
    tokens = {"n": 0}
    prompt = lambda _p: ""

    assert _dispatch_command("/snapshot m1", ctx, console, prompt, last, tokens) is True
    assert (ctx.snapshots.root / "named" / "m1.json").exists()

    assert _dispatch_command("/restore", ctx, console, prompt, last, tokens) is True
    assert _dispatch_command("/restore last", ctx, console, prompt, last, tokens) is True
    assert (ws / "a.txt").read_text(encoding="utf-8") == "v1"

    # 未知快照：报错不崩溃
    assert _dispatch_command("/restore 不存在", ctx, console, prompt, last, tokens) is True
    store = ctx.store
    store.close()


def test_restore_digit_confirms(tmp_path) -> None:
    """/restore <编号> 先预览确认；y 才执行。"""
    from vgent.cli import _dispatch_command

    ctx, ws = _cli_ctx(tmp_path)
    _write(ws, "a.txt", "v1")
    ctx.snapshots.begin_turn()
    ctx.snapshots.note_before_write("a.txt")
    _write(ws, "a.txt", "v2")
    ctx.snapshots.seal_turn()
    console = Console()
    last = tmp_path / "last"
    tokens = {"n": 0}

    # 取消
    assert (
        _dispatch_command("/restore 1", ctx, console, lambda _p: "n", last, tokens) is True
    )
    assert (ws / "a.txt").read_text(encoding="utf-8") == "v2"
    # 确认
    assert (
        _dispatch_command("/restore 1", ctx, console, lambda _p: "y", last, tokens) is True
    )
    assert (ws / "a.txt").read_text(encoding="utf-8") == "v1"
    ctx.store.close()


def test_snapshot_command_no_store(tmp_path) -> None:
    """无快照 store 时命令提示未启用，不崩溃。"""
    from vgent.cli import _dispatch_command

    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=SimpleNamespace(chat=lambda *a, **k: None),
    )
    console = Console()
    assert (
        _dispatch_command("/snapshot", ctx, console, lambda _p: "", tmp_path / "last", {"n": 0})
        is True
    )
    assert (
        _dispatch_command("/restore last", ctx, console, lambda _p: "", tmp_path / "last", {"n": 0})
        is True
    )
    store.close()
