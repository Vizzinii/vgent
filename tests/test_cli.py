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


def test_reflect_inline_short_session(tmp_path) -> None:
    """M7：空会话 /reflect 提示无需反思，不崩溃。"""
    from types import SimpleNamespace

    from vgent.agent import SessionContext
    from vgent.cli import _reflect_inline

    store = _store(tmp_path)
    sid = store.create_session()
    ctx = SessionContext(session_id=sid, store=store, llm=SimpleNamespace(chat=lambda *a, **k: None))
    _reflect_inline(ctx, Console())
    store.close()


def test_reflect_inline_no_llm_output_not_persisted(tmp_path) -> None:
    """M7：有历史但 LLM 无响应：提示未产出内容，不落库不崩溃。"""
    from types import SimpleNamespace

    from vgent.agent import SessionContext
    from vgent.cli import _reflect_inline
    from vgent.messages import Message

    store = _store(tmp_path)
    sid = store.create_session()
    store.add_message(sid, Message("user", "任务"))
    store.add_message(sid, Message("tool", "exit 1\nboom", tool_call_id="c1"))
    ctx = SessionContext(
        session_id=sid, store=store, llm=SimpleNamespace(chat=lambda *a, **k: None)
    )
    _reflect_inline(ctx, Console())
    assert len(store.get_history(sid)) == 2  # 反思未产出 → 不落库
    store.close()


# -- M8：记忆命令处理器 ---------------------------------------------------------


def test_remember_inline_stores_summary(tmp_path) -> None:
    from types import SimpleNamespace

    from vgent.agent import SessionContext
    from vgent.cli import _remember_inline
    from vgent.memory.episodic import EpisodicMemory
    from vgent.messages import Message

    store = _store(tmp_path)
    sid = store.create_session(title="重构任务")
    store.add_message(sid, Message("user", "帮我重构"))
    store.add_message(sid, Message("assistant", "完成"))
    mem = EpisodicMemory(tmp_path / "m.jsonl")

    class FakeLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            return SimpleNamespace(
                messages=[Message("assistant", "做了重构，得出结论：OK，遗留事项是补测试")]
            )

    ctx = SessionContext(session_id=sid, store=store, llm=FakeLLM(), memory=mem)
    _remember_inline(ctx, Console(), "重构")
    assert mem.count() == 1
    assert mem.search("重构")[0].summary == "做了重构，得出结论：OK，遗留事项是补测试"


def test_remember_inline_short_session(tmp_path) -> None:
    from types import SimpleNamespace

    from vgent.agent import SessionContext
    from vgent.cli import _remember_inline
    from vgent.memory.episodic import EpisodicMemory

    store = _store(tmp_path)
    sid = store.create_session()
    mem = EpisodicMemory(tmp_path / "m.jsonl")
    ctx = SessionContext(
        session_id=sid, store=store, llm=SimpleNamespace(chat=lambda *a, **k: None), memory=mem
    )
    _remember_inline(ctx, Console(), "主题")
    assert mem.count() == 0  # 太短不存
    store.close()


def test_recall_inline_injects_and_persists(tmp_path) -> None:
    from types import SimpleNamespace

    from vgent.agent import SessionContext
    from vgent.cli import _recall_inline
    from vgent.memory.episodic import EpisodicMemory

    store = _store(tmp_path)
    sid = store.create_session()
    mem = EpisodicMemory(tmp_path / "m.jsonl")
    mem.add("git 仓库优化", "扫描发现 1 个瓶颈", "other_sid", "优化 git")
    ctx = SessionContext(
        session_id=sid, store=store, llm=SimpleNamespace(chat=lambda *a, **k: None), memory=mem
    )
    _recall_inline(ctx, Console(), "git")
    hist = store.get_history(sid)
    assert any(m.role == "system" and m.content.startswith("[记忆]") for m in hist)
    store.close()


def test_recall_inline_no_match(tmp_path) -> None:
    from types import SimpleNamespace

    from vgent.agent import SessionContext
    from vgent.cli import _recall_inline
    from vgent.memory.episodic import EpisodicMemory

    store = _store(tmp_path)
    sid = store.create_session()
    mem = EpisodicMemory(tmp_path / "m.jsonl")
    mem.add("git 仓库优化", "摘要", "other_sid", "优化 git")
    ctx = SessionContext(
        session_id=sid, store=store, llm=SimpleNamespace(chat=lambda *a, **k: None), memory=mem
    )
    _recall_inline(ctx, Console(), "zzz")
    assert all("[记忆]" not in m.content for m in store.get_history(sid))
    store.close()


def test_memories_inline_empty_and_list(tmp_path) -> None:
    from types import SimpleNamespace

    from vgent.agent import SessionContext
    from vgent.cli import _memories_inline
    from vgent.memory.episodic import EpisodicMemory

    store = _store(tmp_path)
    sid = store.create_session()
    mem = EpisodicMemory(tmp_path / "m.jsonl")
    mem.add("重构", "做了重构", "other_sid", "重构任务")
    ctx = SessionContext(
        session_id=sid, store=store, llm=SimpleNamespace(chat=lambda *a, **k: None), memory=mem
    )
    _memories_inline(ctx, Console())  # 有记忆：打印不崩溃
    ctx2 = SessionContext(
        session_id=sid, store=store, llm=SimpleNamespace(chat=lambda *a, **k: None),
        memory=EpisodicMemory(tmp_path / "empty.jsonl"),
    )
    _memories_inline(ctx2, Console())  # 空记忆：提示不崩溃
    store.close()


# -- M10：外部命令分发 ------------------------------------------------------------


def test_dispatch_external_command(tmp_path) -> None:
    """M10：外部命令命中 → run(ctx, args) 执行并返回 True；普通文本返回 False。"""
    from types import SimpleNamespace

    from vgent.agent import SessionContext
    from vgent.cli import _dispatch_command

    store = _store(tmp_path)
    sid = store.create_session()
    seen: list[tuple[str, str]] = []

    def mycmd(ctx, args: str) -> str:
        seen.append((args, "called"))
        return "外部命令输出"

    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=SimpleNamespace(chat=lambda *a, **k: None),
        ext_commands={"mycmd": mycmd},
    )
    assert _dispatch_command("/mycmd hello", ctx, Console(), lambda p: "", tmp_path / "last", {"n": 0}) is True
    assert seen == [("hello", "called")]
    # 普通文本（非 /）交给 LLM
    assert _dispatch_command("帮我写代码", ctx, Console(), lambda p: "", tmp_path / "last", {"n": 0}) is False
    # 未注册的外部命令名 → 交给 LLM
    assert _dispatch_command("/nosuch", ctx, Console(), lambda p: "", tmp_path / "last", {"n": 0}) is False
    store.close()


def test_dispatch_builtin_priority_over_external(tmp_path) -> None:
    """M10：外部命令与内置重名 → 内置优先（外部不执行）。"""
    from types import SimpleNamespace

    from vgent.agent import SessionContext
    from vgent.cli import _dispatch_command

    store = _store(tmp_path)
    sid = store.create_session()
    called = False

    def evil(ctx, args: str) -> str:
        nonlocal called
        called = True
        return "不应执行"

    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=SimpleNamespace(chat=lambda *a, **k: None),
        ext_commands={"list": evil},
    )
    assert _dispatch_command("/list", ctx, Console(), lambda p: "", tmp_path / "last", {"n": 0}) is True
    assert called is False  # 内置 /list 优先，外部未执行
    store.close()


def test_dispatch_external_error_not_crash(tmp_path) -> None:
    """M10：外部命令抛异常 → 捕获提示，不崩溃。"""
    from types import SimpleNamespace

    from vgent.agent import SessionContext
    from vgent.cli import _dispatch_command

    store = _store(tmp_path)
    sid = store.create_session()

    def boom(ctx, args: str) -> str:
        raise RuntimeError("坏了")

    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=SimpleNamespace(chat=lambda *a, **k: None),
        ext_commands={"boom": boom},
    )
    assert _dispatch_command("/boom", ctx, Console(), lambda p: "", tmp_path / "last", {"n": 0}) is True
    store.close()
