"""M12 测试方案 P0 补测（永久保留）：上下文预算 / 快照恢复 / 记忆管线的缺口项。

来源：M12 测试方案缺口表 G-A1..G-C9（FakeLLM / tmp_path，不触网）。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from vgent.agent import SessionContext, _persist_compacted, run_turn
from vgent.config import Config, ContextConfig, load_config
from vgent.context import ContextEngine, _tiktoken_count
from vgent.llm import ChatResult
from vgent.memory.pipeline import (
    MemoryPipeline,
    RoundContent,
    _combine_rounds,
    make_pipeline_for_workspace,
)
from vgent.memory.store import MEMORY_NAME, MemoryFileStore
from vgent.messages import Message, ToolCall, Usage
from vgent.permission import ConfirmResult, PermissionSystem
from vgent.snapshot import MAX_FILE_BYTES, SnapshotStore
from vgent.store import SessionStore
from vgent.tools import ToolRegistry, ToolSchema

STAGE2_REWRITE = (
    '{"unchanged": false, "MEMORY_md": "v1\\n# MEMORY\\n\\n## 决策\\n- 用 Redis 7 做二级缓存\\n", '
    '"memory_summary_md": "v1\\n# Memory Summary\\n\\nRedis 7 二级缓存\\n"}'
)
STAGE1_JSON = '{"raw_bullets": ["事实X"], "rollout_summary": ""}'


# =============================================================================
# 上下文预算 + compact 持久化
# =============================================================================


def _final(text: str = "好") -> ChatResult:
    return ChatResult(messages=[Message("assistant", text)], usage=Usage(10, 5, 15))


class Scripted:
    def __init__(self, responses: list[ChatResult]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def test_tiktoken_real_count_and_unknown_model_fallback() -> None:
    """G-A1：tiktoken 真实计数可用；未知模型回退 cl100k_base 不抛。"""
    n = _tiktoken_count("hello world")
    assert n is not None and n > 0
    n2 = _tiktoken_count("你好世界" * 30, model="deepseek-v4-flash")
    assert n2 is not None and n2 > 0


def test_tiktoken_path_smaller_than_heuristic() -> None:
    """G-A1：连续重复字符下 tiktoken 压缩率优于 len//3 启发式（真实路径生效）。"""
    engine = ContextEngine()
    msgs = [Message("user", "x" * 2000)]
    assert engine.estimate_send_tokens(msgs) < engine._estimate_tokens(msgs)


def test_config_parses_reserved_and_light_model(tmp_path) -> None:
    """G-A2：config.toml 解析 reserved_output_tokens 与 light_model。"""
    p = tmp_path / "config.toml"
    p.write_text(
        '[provider]\nactive = "p1"\n\n'
        '[providers.p1]\nbase_url = "http://x"\napi_key = "k"\n'
        'model = "m1"\nlight_model = "m2"\ncontext_length = 12345\n\n'
        "[context]\nreserved_output_tokens = 999\n",
        encoding="utf-8",
    )
    cfg = load_config(path=p)
    assert cfg.provider.light_model == "m2"
    assert cfg.provider.model == "m1"
    assert cfg.context.reserved_output_tokens == 999


def test_config_defaults_light_model_and_reserved(tmp_path) -> None:
    """G-A2：缺省 light_model 空串、reserved_output_tokens 0。"""
    p = tmp_path / "config.toml"
    p.write_text(
        '[provider]\nactive = "p1"\n\n'
        '[providers.p1]\nbase_url = "http://x"\napi_key = "k"\nmodel = "m1"\n',
        encoding="utf-8",
    )
    cfg = load_config(path=p)
    assert cfg.provider.light_model == ""
    assert cfg.context.reserved_output_tokens == 0


def test_compact_inline_persists_and_rebuilds(tmp_path) -> None:
    """G-A3：/compact 落库 → 新会话 run_turn 底稿 = 头部+摘要+保留尾部（不发全量）。"""
    from vgent.cli import _compact_inline

    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    for i in range(10):
        store.add_message(sid, Message("user", f"m{i}" + "x" * 40))
    engine = ContextEngine(context_length=100_000, cfg=ContextConfig(tail_token_budget=60))
    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=Scripted([]),  # /compact 不调 LLM
        engine=engine,
        data_dir=tmp_path,
    )
    _compact_inline(ctx, Console())
    comp = store.get_compact(sid)
    assert comp is not None
    summary, retained, _boundary = comp
    assert "TailWindow" in summary and retained

    llm2 = Scripted([_final()])
    ctx2 = SessionContext(
        session_id=sid,
        store=store,
        llm=llm2,
        engine=ContextEngine(context_length=100_000, cfg=ContextConfig(tail_token_budget=60)),
    )
    run_turn("继续", ctx2)
    sent = "\n".join(m.content for m in llm2.calls[0])
    assert "m0" in sent  # 头部保留
    assert "TailWindow" in sent  # 摘要消息
    assert "m9" in sent  # 保留尾部
    assert "m5" not in sent  # 中间被压缩，不再全量重发
    assert llm2.calls[0][-1].content == "继续"
    store.close()


def test_persist_compacted_noop_and_overwrite(tmp_path) -> None:
    """G-A4/G-A5：无变化不写；连续压缩覆盖只留最新。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    store.add_message(sid, Message("user", "a"))
    store.add_message(sid, Message("user", "b"))
    ctx = SessionContext(session_id=sid, store=store, llm=SimpleNamespace())
    msgs = store.get_history(sid)
    _persist_compacted(ctx, msgs, msgs)  # after is before → 不写
    assert store.get_compact(sid) is None
    _persist_compacted(ctx, msgs, [msgs[0], Message("system", "S1"), msgs[1]])
    _persist_compacted(ctx, msgs, [msgs[0], Message("system", "S2"), msgs[1]])
    comp = store.get_compact(sid)
    assert comp is not None and comp[0] == "S2"
    store.close()


def test_web_compact_persists(tmp_path, monkeypatch) -> None:
    """G-A6：web /compact 落库。"""
    from vgent.web.server import HubManager, run_command

    monkeypatch.chdir(tmp_path)
    cfg = Config(data_dir=tmp_path)
    cfg.context.tail_token_budget = 60
    store = SessionStore(tmp_path / "t.db")
    mgr = HubManager(cfg, store, Scripted([]))
    sid = store.create_session()
    for _ in range(10):
        store.add_message(sid, Message("user", "x" * 40))
    hub = mgr.hub(sid)
    out = run_command("/compact", hub)
    assert "已压缩" in out
    assert hub.ctx.store.get_compact(sid) is not None
    store.close()


# =============================================================================
# 快照/恢复
# =============================================================================


def _snap(tmp_path):
    cwd = tmp_path / "ws"
    cwd.mkdir(exist_ok=True)
    return SnapshotStore(tmp_path / "ckpt" / "sess1", cwd), cwd


def _write(cwd: Path, rel: str, content: str) -> None:
    p = cwd / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_restore_named_missing_raises(tmp_path) -> None:
    """G-B3：restore_named 不存在抛 FileNotFoundError。"""
    store, _ = _snap(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.restore_named("不存在")


def test_restore_skips_directory_and_escape_and_too_large(tmp_path) -> None:
    """G-B4：目录 / 越界 / 超大文件 → skip 不落盘不删文件。"""
    store, cwd = _snap(tmp_path)
    (cwd / "d").mkdir()
    big = cwd / "big.txt"
    big.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
    store.begin_turn()
    store.note_before_write("d")
    store.note_before_write("../escape.txt")
    store.note_before_write("big.txt")
    store.seal_turn()
    report = store.restore_last()
    reasons = {rel: why for rel, why in report.skipped}
    assert reasons["d"] == "is_directory"
    assert "out of workspace" in reasons["../escape.txt"]
    assert reasons["big.txt"] == "too_large"
    assert report.restored == [] and report.deleted == []


def test_version_increments_across_turns(tmp_path) -> None:
    """G-B5：跨回合版本号递增 1→2。"""
    store, cwd = _snap(tmp_path)
    _write(cwd, "a.txt", "v0")
    store.begin_turn()
    store.note_before_write("a.txt")
    _write(cwd, "a.txt", "v1")
    store.seal_turn()
    store.begin_turn()
    store.note_before_write("a.txt")
    _write(cwd, "a.txt", "v2")
    store.seal_turn()
    items = store._snapshot_items()
    assert items[0]["files"]["a.txt"]["version"] == 1
    assert items[1]["files"]["a.txt"]["version"] == 2


def test_named_captures_all_session_files(tmp_path) -> None:
    """G-B6：命名档拍全量（多文件）后 restore_named 双文件还原。"""
    store, cwd = _snap(tmp_path)
    _write(cwd, "a.txt", "1")
    _write(cwd, "sub/b.txt", "2")
    store.begin_turn()
    store.note_before_write("a.txt")
    store.note_before_write("sub/b.txt")
    store.seal_turn()
    name = store.save_named("全量")
    report = store.restore_named(name)
    assert sorted(report.restored) == ["a.txt", "sub/b.txt"]


def test_delete_session_keeps_checkpoints_dir(tmp_path) -> None:
    """G-B7：delete_session 不删快照目录（设计确认，靠 30 天 GC）。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    ck = tmp_path / "checkpoints" / sid
    ck.mkdir(parents=True)
    (ck / "x").write_text("1", encoding="utf-8")
    store.delete_session(sid)
    assert ck.exists()
    store.close()


def test_restore_undo_via_dispatch(tmp_path) -> None:
    """G-B2：CLI /restore last 后再 /restore undo 回到 restore 前状态。"""
    from vgent.cli import _dispatch_command

    ws = tmp_path / "ws"
    ws.mkdir()
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=SimpleNamespace(chat=lambda *a, **k: None),
        data_dir=tmp_path,
    )
    ctx.snapshots = SnapshotStore(tmp_path / "ckpt" / sid, ws)
    _write(ws, "a.txt", "v1")
    ctx.snapshots.begin_turn()
    ctx.snapshots.note_before_write("a.txt")
    _write(ws, "a.txt", "v2")
    ctx.snapshots.seal_turn()
    console = Console()
    last = tmp_path / "last"
    tokens = {"n": 0}
    assert _dispatch_command("/restore last", ctx, console, lambda p: "", last, tokens) is True
    assert (ws / "a.txt").read_text(encoding="utf-8") == "v1"
    assert _dispatch_command("/restore undo", ctx, console, lambda p: "", last, tokens) is True
    assert (ws / "a.txt").read_text(encoding="utf-8") == "v2"
    store.close()


# =============================================================================
# 记忆管线
# =============================================================================


def _round(workspace: Path, user: str = "优化 git 仓库性能", **kw) -> RoundContent:
    return RoundContent(
        workspace=str(workspace.resolve()), session_id="s1", user_text=user, **kw
    )


def _mstore(tmp_path) -> MemoryFileStore:
    return MemoryFileStore(tmp_path / "data", tmp_path / "ws")


class AdaptiveLLM:
    """stage1/stage2 按 system 内容区分返回（线程下调用次数不定也稳）。"""

    def __init__(self, stage1: str, stage2: str) -> None:
        self.stage1 = stage1
        self.stage2 = stage2

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        system = next(
            (m.content for m in messages if getattr(m, "role", None) == "system"), ""
        )
        if "抽取器" in system:
            return SimpleNamespace(messages=[Message("assistant", self.stage1)])
        return SimpleNamespace(messages=[Message("assistant", self.stage2)])


def test_stage1_empty_result_no_write(tmp_path) -> None:
    """G-C3：stage1 空结果不写 raw、无 pending。"""
    store = _mstore(tmp_path)
    llm = SimpleNamespace(
        chat=lambda *a, **k: SimpleNamespace(
            messages=[Message("assistant", '{"raw_bullets": [], "rollout_summary": ""}')]
        )
    )
    pipe = MemoryPipeline(store, llm, "m")
    pipe._process_batch([_round(tmp_path)])
    assert pipe.pending_signal_count == 0
    raw = (store.root / "raw_memories.md").read_text(encoding="utf-8")
    assert raw.strip() == "# raw_memories"


def test_stage1_llm_exception_silent(tmp_path) -> None:
    """G-C4：stage1 LLM 异常静默（不抛不写）。"""
    store = _mstore(tmp_path)

    class Boom:
        def chat(self, *a, **k):
            raise RuntimeError("llm down")

    pipe = MemoryPipeline(store, Boom(), "m")
    pipe._process_batch([_round(tmp_path)])
    assert pipe.pending_signal_count == 0
    raw = (store.root / "raw_memories.md").read_text(encoding="utf-8")
    assert "事实" not in raw


def test_blacklisted_rollout_summary_dropped(tmp_path) -> None:
    """G-C5：rollout_summary 命中黑名单 → 置空，不写 rollout 行。"""
    store = _mstore(tmp_path)
    llm = SimpleNamespace(
        chat=lambda *a, **k: SimpleNamespace(
            messages=[
                Message(
                    "assistant",
                    '{"raw_bullets": ["结论：升级"], "rollout_summary": "api_key = secret"}',
                )
            ]
        )
    )
    pipe = MemoryPipeline(store, llm, "m")
    pipe._process_batch([_round(tmp_path)])
    raw = (store.root / "raw_memories.md").read_text(encoding="utf-8")
    assert "结论：升级" in raw
    assert "secret" not in raw
    assert "rollout:" not in raw


def test_drain_keeps_pending_on_parse_failure(tmp_path) -> None:
    """G-C6：drain 遇 stage2 持续解析失败 → 保留 pending（有界重试不挂死）。"""
    store = _mstore(tmp_path)

    class Bad2:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            self.calls += 1
            system = next(
                (m.content for m in messages if getattr(m, "role", None) == "system"), ""
            )
            if "抽取器" in system:
                return SimpleNamespace(messages=[Message("assistant", STAGE1_JSON)])
            return SimpleNamespace(messages=[Message("assistant", "garbage output")])

    pipe = MemoryPipeline(store, Bad2(), "m")
    pipe.submit(_round(tmp_path))
    pipe.drain()
    assert pipe.pending_signal_count == 1
    assert pipe.pending_count == 0


def test_memory_summary_injected_every_turn_and_none_noop(tmp_path) -> None:
    """G-C7：memory_summary None 不注入；有值每轮注入（含第二轮）。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()

    llm = Scripted([_final()])
    ctx = SessionContext(session_id=sid, store=store, llm=llm)
    run_turn("问", ctx)
    assert not any(m.role == "system" and "记忆总览" in m.content for m in llm.calls[0])

    llm2 = Scripted([_final(), _final()])
    ctx2 = SessionContext(
        session_id=sid, store=store, llm=llm2, memory_summary="Redis 7 二级缓存"
    )
    run_turn("继续", ctx2)
    run_turn("再继续", ctx2)
    for call in llm2.calls:
        assert any(m.role == "system" and "记忆总览" in m.content for m in call)
    store.close()


def test_combine_rounds_merges_multiple(tmp_path) -> None:
    """G-C8：_combine_rounds 合并多轮文本。"""
    text = _combine_rounds(
        [
            _round(tmp_path, user="a"),
            _round(tmp_path, user="b"),
            _round(tmp_path, user="c"),
        ]
    )
    assert "轮次 1" in text and "轮次 3" in text


def test_submit_three_rounds_consolidates(tmp_path) -> None:
    """G-C8：线程路径 3 信号触发 stage2 落盘 MEMORY/summary。"""
    store = _mstore(tmp_path)
    pipe = MemoryPipeline(store, AdaptiveLLM(STAGE1_JSON, STAGE2_REWRITE), "m")
    for i in range(3):
        pipe.submit(_round(tmp_path, user=f"第{i}轮"))
    pipe.drain()
    assert pipe.pending_signal_count == 0
    assert "Redis 7" in store.read_rel(MEMORY_NAME, limit=None)


def test_pipeline_uses_given_model(tmp_path) -> None:
    """G-C9：make_pipeline_for_workspace 透传指定 model（light_model 接线）。"""
    client = SimpleNamespace(
        chat=lambda *a, **k: SimpleNamespace(messages=[Message("assistant", "x")])
    )
    pipe = make_pipeline_for_workspace(
        tmp_path / "data", tmp_path / "ws", client, model="light-model"
    )
    assert pipe._model == "light-model"


def test_cli_memory_inline_subcommands(tmp_path) -> None:
    """G-C1：CLI /memory 子命令族（show/path/grep/help/未知/clear/无 store）。"""
    from vgent.cli import _memory_inline

    data_home = tmp_path / "data"
    ws = tmp_path / "ws"
    store = MemoryFileStore(data_home, ws)
    store.ensure_layout()
    store.atomic_write(
        MEMORY_NAME, "v1\n# MEMORY\n\n## 缓存\n- Redis 7 TTL 300（二级缓存方案）\n"
    )
    sess = SessionStore(tmp_path / "t.db")
    ctx = SessionContext(
        session_id="s1",
        store=sess,
        llm=SimpleNamespace(chat=lambda *a, **k: None),
        memory_file_store=store,
        memory_pipeline=SimpleNamespace(
            pending_count=1, pending_signal_count=2, invalidate=lambda: None
        ),
    )
    console = Console()
    _memory_inline(ctx, console, "")
    _memory_inline(ctx, console, "show")
    _memory_inline(ctx, console, "path")
    _memory_inline(ctx, console, "grep Redis")
    _memory_inline(ctx, console, "help")
    _memory_inline(ctx, console, "badcmd")
    _memory_inline(ctx, console, "clear")
    assert store.summary_is_placeholder()
    ctx2 = SessionContext(
        session_id="s2", store=sess, llm=SimpleNamespace(chat=lambda *a, **k: None)
    )
    _memory_inline(ctx2, console, "")
    sess.close()


def test_web_memory_cmd(tmp_path, monkeypatch) -> None:
    """G-C2：web /memory 命令族。"""
    from vgent.web.server import HubManager, run_command

    monkeypatch.chdir(tmp_path)
    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    mgr = HubManager(cfg, store, Scripted([]))
    sid = store.create_session()
    hub = mgr.hub(sid)
    out = run_command("/memory", hub)
    assert "记忆" in out
    out = run_command("/memory show", hub)
    assert isinstance(out, str)
    out = run_command("/memory path", hub)
    assert "memory" in out
    out = run_command("/memory clear", hub)
    assert "已清空" in out
    out = run_command("/memory badcmd", hub)
    assert "未知" in out
    store.close()


# =============================================================================
# M12 中断修复：Ctrl-C 落在工具执行窗口 → 孤儿 assistant(tool_calls) 清洗
# =============================================================================


def test_interrupt_orphan_reproduced(tmp_path) -> None:
    """复现：工具执行中抛 KeyboardInterrupt → 库里留下孤儿 assistant(tool_calls)（无 tool 响应）。"""
    from vgent.agent import run_turn

    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    llm = Scripted(
        [
            ChatResult(
                messages=[Message("assistant", "", tool_calls=[ToolCall("c1", "boom", "{}")])],
                usage=Usage(10, 5, 15),
                tool_calls=[ToolCall("c1", "boom", "{}")],
            )
        ]
    )

    class Boom:
        def handler(self, args):
            raise KeyboardInterrupt("模拟 Ctrl-C 落在工具执行")

    registry = ToolRegistry()
    registry.register(
        ToolSchema(name="boom", description="中断用工具", parameters={}, permission="exec"),
        Boom().handler,
    )
    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=llm,
        tools=registry,
        permissions=PermissionSystem(confirm=lambda t, a: ConfirmResult.APPROVE),
    )
    with pytest.raises(KeyboardInterrupt):
        run_turn("跑一下", ctx)
    # 复现毒化状态：历史末尾是带 tool_calls 的 assistant，且存在未兑现的 tool_call_id
    hist = store.get_history(sid)
    assert hist[-1].role == "assistant" and hist[-1].tool_calls is not None
    declared = {
        tc.id for m in hist if m.role == "assistant" and m.tool_calls for tc in m.tool_calls
    }
    received = {m.tool_call_id for m in hist if m.role == "tool"}
    assert declared - received  # 有孤儿（无 tool 响应）
    store.close()


def test_interrupt_orphan_repaired_on_next_turn(tmp_path) -> None:
    """修复验证：毒化会话再 run_turn → 发送列表不再含悬空 tool_calls（孤儿被清洗，API 不 400）。"""
    from vgent.agent import run_turn

    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    store.add_message(sid, Message("user", "跑一下"))
    store.add_message(sid, Message("assistant", "", tool_calls=[ToolCall("c1", "boom", "{}")]))
    llm = Scripted([_final("好的")])
    ctx = SessionContext(session_id=sid, store=store, llm=llm, tools=ToolRegistry())
    run_turn("继续", ctx)
    sent = llm.calls[0]
    leaked = [m for m in sent if m.role == "assistant" and m.tool_calls]
    assert not leaked  # 悬空 tool_calls 不进发送列表
    # SQLite 全量历史不动（孤儿仍留在库里，作为中断事实的记录）
    hist = store.get_history(sid)
    orphan = [m for m in hist if m.role == "assistant" and m.tool_calls]
    assert len(orphan) == 1  # 孤儿保留在库（未被修复删改），只是不进发送列表
    store.close()
