"""批 1 回归测试（评审修复 F1-F5，见 HANDOFF「修复计划 · 评审修复批 1/2」）。

F1 /compact 后同进程陈旧快照丢中间轮（P0，修复前已复现）
F2 edit_file 对非 UTF-8 文件有损往返损坏原文（P0，修复前已复现：GBK → 锟斤拷）
F3 Ctrl-C（KeyboardInterrupt）打死整个 REPL
F4 memory_summary 启动快照陈旧（管线重写文件后注入旧值）
F5 rollout 文件名同秒撞车静默覆盖
"""
from __future__ import annotations

import io

from rich.console import Console

from vgent.agent import SessionContext, run_turn
from vgent.cli import _repl
from vgent.context import ContextEngine
from vgent.llm import ChatResult
from vgent.memory.store import SUMMARY_NAME, MemoryFileStore
from vgent.messages import Message, Usage
from vgent.store import SessionStore
from vgent.tools import default_tools


class FakeLLM:
    def __init__(self, reply: str = "回复内容") -> None:
        self.calls: list[list[Message]] = []
        self.reply = reply

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(list(messages))
        if on_delta:
            on_delta(self.reply)
        return ChatResult(
            messages=[Message("assistant", self.reply)], usage=Usage(10, 5, 15)
        )


def _ctx(tmp_path, llm, engine=None, **kw) -> tuple[SessionContext, SessionStore]:
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    ctx = SessionContext(
        session_id=sid, store=store, llm=llm, engine=engine or ContextEngine(), **kw
    )
    return ctx, store


def test_compacted_stale_snapshot_fixed(tmp_path) -> None:
    """F1：设 engine.compacted（模拟 /compact，含落库）后连续两轮，
    第二轮 send 必须含第一轮的 user 与 assistant 消息。

    修复前：engine.compacted 只在 /new、/resume 清空，第二轮 send 仍是旧快照，
    FIVE 整轮对模型不可见（SQLite 在、发送列表缺）。
    """
    llm = FakeLLM()
    ctx, store = _ctx(tmp_path, llm)
    run_turn("ONE", ctx)
    run_turn("TWO", ctx)
    # 模拟 /compact：底稿 = [头, 摘要, 尾]，并按 _compact_inline 同口径落库
    msgs = store.get_history(ctx.session_id)
    summary = "【历史摘要（原 3 条）】要点"
    boundary = store.last_message_id(ctx.session_id)
    store.upsert_compact(ctx.session_id, summary, [msgs[-1]], boundary)
    ctx.engine.compacted = [msgs[0], Message("system", summary), msgs[-1]]

    run_turn("FIVE", ctx)
    run_turn("SIX", ctx)

    six = llm.calls[-1]
    assert any(m.role == "user" and m.content == "FIVE" for m in six)
    assert any(m.role == "assistant" and m.content == "回复内容" for m in six)
    # 第一轮（FIVE 轮）本身仍以底稿为基线（含摘要）
    five = llm.calls[-2]
    assert any("历史摘要" in (m.content or "") for m in five)
    store.close()


def test_edit_file_rejects_non_utf8(tmp_path) -> None:
    """F2：GBK 字节文件 edit → 拒绝编辑 + 文件字节原样（不再有损写回）。"""
    p = tmp_path / "gbk.txt"
    raw = "中文内容 abc".encode("gbk")
    p.write_bytes(raw)
    out = default_tools().execute(
        "edit_file",
        {"path": str(p), "old_string": "abc", "new_string": "xyz"},
    )
    assert "拒绝编辑" in out
    assert "非 UTF-8" in out
    assert p.read_bytes() == raw  # 未被写坏
    # UTF-8 文件正常编辑不受影响
    p2 = tmp_path / "ok.txt"
    p2.write_text("hello abc\n", encoding="utf-8")
    out2 = default_tools().execute(
        "edit_file", {"path": str(p2), "old_string": "abc", "new_string": "xyz"}
    )
    assert "已替换 1 处" in out2
    assert p2.read_text(encoding="utf-8") == "hello xyz\n"


def test_repl_keyboard_interrupt_survives(tmp_path, monkeypatch) -> None:
    """F3：run_turn 中 Ctrl-C 只中断当轮，REPL 继续接受输入（不退出进程）。"""
    import vgent.cli as cli_mod

    ctx, _store = _ctx(tmp_path, FakeLLM())

    def fake_run_turn(text, ctx, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod, "run_turn", fake_run_turn)
    inputs = iter(["第一轮", "/exit"])
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False)
    # 若 KeyboardInterrupt 穿透（修复前行为），这里会直接抛出而不是正常返回
    _repl(ctx, console, lambda ps: next(inputs), tmp_path / "last", {"n": 0})
    assert "已中断本轮" in buf.getvalue()


def test_memory_summary_refreshed_each_turn(tmp_path) -> None:
    """F4：memory_file_store 在场时每轮从文件重读——管线重写 summary 后
    下一轮注入的是新内容，不是启动快照。"""
    llm = FakeLLM()
    mstore = MemoryFileStore(tmp_path / "home", tmp_path / "ws")
    ctx, store = _ctx(tmp_path, llm, memory_file_store=mstore)

    run_turn("第一问", ctx)  # 空（placeholder）summary → 不注入
    assert not any("记忆总览" in (m.content or "") for m in llm.calls[0])

    mstore.atomic_write(SUMMARY_NAME, "v1\n# Memory Summary\n\n新要点：Redis 7 二级缓存。\n")
    run_turn("第二问", ctx)
    hit = [m for m in llm.calls[1] if m.role == "system" and "记忆总览" in m.content]
    assert hit and "Redis 7" in hit[0].content
    store.close()


def test_write_rollout_unique_same_second(tmp_path) -> None:
    """F5：同 session 同秒两次 write_rollout 生成两个文件，不静默覆盖。"""
    mstore = MemoryFileStore(tmp_path / "home", tmp_path / "ws")
    mstore.ensure_layout()
    rel1 = mstore.write_rollout("sess-1", "# A\n")
    rel2 = mstore.write_rollout("sess-1", "# B\n")
    assert rel1 != rel2
    assert (mstore.root / rel1).is_file()
    assert (mstore.root / rel2).is_file()
    assert "# A" in (mstore.root / rel1).read_text(encoding="utf-8")
    assert "# B" in (mstore.root / rel2).read_text(encoding="utf-8")
