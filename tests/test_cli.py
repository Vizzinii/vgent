"""M4 前置（zcode 化）测试：CLI flag 族 + 记住上次会话。"""
from __future__ import annotations

from rich.console import Console

from vgent.cli import (
    _build_parser,
    _headless,
    _last_session,
    _last_session_path,
    _remember_session,
    _resolve_resume,
    _resolve_start_session,
)
from vgent.config import Config
from vgent.store import SessionStore


def _store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "t.db")


def test_list_sessions_flag(tmp_path) -> None:
    store = _store(tmp_path)
    store.create_session(title="a")
    args = _build_parser().parse_args(["--list-sessions"])
    assert _headless(store, args, Console()) == 0
    store.close()


def test_delete_session_flag(tmp_path) -> None:
    store = _store(tmp_path)
    sid = store.create_session(title="a")
    args = _build_parser().parse_args(["--delete-session", sid])
    assert _headless(store, args, Console()) == 0
    assert store.list_sessions() == []
    args2 = _build_parser().parse_args(["--delete-session", "no-such"])
    assert _headless(store, args2, Console()) == 1
    store.close()


def test_remember_last_session(tmp_path) -> None:
    store = _store(tmp_path)
    path = _last_session_path(Config(data_dir=tmp_path))
    assert _last_session(store, path) is None
    sid = store.create_session(title="a")
    _remember_session(path, sid)
    assert _last_session(store, path) == sid
    # 会话被删后 last 失效
    store.delete_session(sid)
    assert _last_session(store, path) is None
    store.close()


def test_resolve_resume(tmp_path) -> None:
    store = _store(tmp_path)
    path = _last_session_path(Config(data_dir=tmp_path))
    s1 = store.create_session(title="first")
    s2 = store.create_session(title="second")
    # last：优先上次会话
    _remember_session(path, s1)
    assert _resolve_resume(store, "last", path) == s1
    # 编号：1 = 最近一个（list 按 created_at 倒序，同秒按 rowid 倒序兜底）
    assert _resolve_resume(store, "1", path) == s2
    assert _resolve_resume(store, "2", path) == s1
    # 会话 id
    assert _resolve_resume(store, s2, path) == s2
    # 无效
    assert _resolve_resume(store, "99", path) is None
    assert _resolve_resume(store, "no-such-id", path) is None
    store.close()


def test_start_session_new_and_resume_flags(tmp_path) -> None:
    store = _store(tmp_path)
    s1 = store.create_session(title="a")
    console = Console()
    path = _last_session_path(Config(data_dir=tmp_path))
    # --new 直接新建
    args = _build_parser().parse_args(["--new"])
    sid = _resolve_start_session(store, args, lambda p: "", console, path)
    assert sid != s1
    assert store.get_title(sid) is not None
    # --resume 缺省：无 last 文件 → 恢复最近一个（即刚新建的）
    args2 = _build_parser().parse_args(["--resume"])
    assert _resolve_start_session(store, args2, lambda p: "", console, path) == sid
    store.close()


def test_compact_inline_sets_compacted(tmp_path) -> None:
    """M4：/compact 处理器把 Summarize 结果设为发送底稿；太短时不设。"""
    from types import SimpleNamespace

    from vgent.agent import SessionContext
    from vgent.cli import _compact_inline
    from vgent.config import ContextConfig
    from vgent.context import ContextEngine
    from vgent.messages import Message

    store = _store(tmp_path)
    sid = store.create_session()
    for i in range(10):
        store.add_message(sid, Message("user", f"第 {i} 条历史内容 xxxxxxxxxx"))
    engine = ContextEngine(
        context_length=1000,
        cfg=ContextConfig(tail_token_budget=30),
        summarizer=lambda middle: "摘要要点",
    )
    dummy_llm = SimpleNamespace(chat=lambda *a, **k: None)
    ctx = SessionContext(session_id=sid, store=store, llm=dummy_llm, engine=engine)

    _compact_inline(ctx, Console())
    assert ctx.engine.compacted is not None
    assert len(ctx.engine.compacted) < 10
    assert any(m.role == "system" and "摘要" in m.content for m in ctx.engine.compacted)

    # 太短（无消息）的会话不设底稿
    short_sid = store.create_session()
    ctx2 = SessionContext(session_id=short_sid, store=store, llm=dummy_llm, engine=ContextEngine())
    _compact_inline(ctx2, Console())
    assert ctx2.engine.compacted is None
    store.close()
