"""M12-C 测试：记忆两阶段管线（MemoryFileStore / prompts / MemoryPipeline / agent 接线 / 工具）。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vgent.agent import SessionContext, run_turn
from vgent.llm import ChatResult
from vgent.memory.episodic import EpisodicMemory, MemoryEntry, memory_note_text
from vgent.memory.pipeline import (
    CONSOLIDATE_MIN_SIGNALS,
    MemoryPipeline,
    RoundContent,
    should_extract,
    slice_round,
)
from vgent.memory.prompts import (
    is_blacklisted,
    parse_json_object,
    parse_stage1,
    parse_stage2,
)
from vgent.memory.store import (
    MEMORY_NAME,
    MemoryFileStore,
)
from vgent.memory.tools import make_memory_tools
from vgent.messages import Message, ToolCall, Usage
from vgent.store import SessionStore
from vgent.tools import ToolRegistry

STAGE2_REWRITE = (
    '{"unchanged": false, "MEMORY_md": "v1\\n# MEMORY\\n\\n## 决策\\n- 用 Redis 7 做二级缓存\\n", '
    '"memory_summary_md": "v1\\n# Memory Summary\\n\\nRedis 7 二级缓存\\n"}'
)


class FakeChatLLM:
    """按调用顺序返回预设文本的假 LLM（管线 stage1/stage2 用，不触网）。"""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(list(messages))
        text = self.responses.pop(0) if self.responses else ""
        return SimpleNamespace(messages=[Message("assistant", text)])


def _round(workspace: Path, user: str = "优化 git 仓库性能", **kw) -> RoundContent:
    return RoundContent(workspace=str(workspace.resolve()), session_id="s1", user_text=user, **kw)


def _store(tmp_path) -> MemoryFileStore:
    return MemoryFileStore(tmp_path / "data", tmp_path / "ws")


# -- prompts：解析与黑名单 ----------------------------------------------------


def test_parse_stage1_extracts_bullets_and_summary() -> None:
    bullets, summary = parse_stage1(
        '{"raw_bullets": ["结论：升级依赖", "用 Redis 做缓存"], "rollout_summary": "本轮确定缓存方案"}'
    )
    assert bullets == ["结论：升级依赖", "用 Redis 做缓存"]
    assert summary == "本轮确定缓存方案"


def test_parse_stage1_empty_and_bad_json() -> None:
    assert parse_stage1("") == ([], "")
    assert parse_stage1("not json at all") == ([], "")
    assert parse_stage1('{"raw_bullets": "not-a-list"}') == ([], "")


def test_parse_stage2_rewrite() -> None:
    unchanged, memory, summary = parse_stage2(STAGE2_REWRITE)
    assert unchanged is False
    assert memory is not None and "Redis 7" in memory
    assert summary is not None and summary.startswith("v1")


def test_parse_stage2_unchanged_and_bad() -> None:
    assert parse_stage2('{"unchanged": true}') == (True, None, None)
    # 缺字段/坏输出：unchanged=False（调用方保留 pending 重试，不消费）
    assert parse_stage2('{"unchanged": false, "MEMORY_md": "x"}') == (False, None, None)
    assert parse_stage2("garbage") == (False, None, None)


def test_parse_json_object_strips_fences() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('前文 {"a": 1} 后文') == {"a": 1}


def test_is_blacklisted_matches_secrets() -> None:
    assert is_blacklisted("sk-abcdef0123456789abcdef 泄露")
    assert is_blacklisted("api_key = xyz123")
    assert is_blacklisted("password: hunter2")
    assert is_blacklisted("token=abc123")
    assert not is_blacklisted("结论：升级依赖到最新版")


# -- store：文件层 ------------------------------------------------------------


def test_ensure_layout_creates_templates(tmp_path) -> None:
    store = _store(tmp_path)
    store.ensure_layout()
    assert store.summary_is_placeholder()
    assert not store.memory_has_entries()
    assert store.read_summary().startswith("v1")


def test_append_raw_and_write_rollout(tmp_path) -> None:
    store = _store(tmp_path)
    store.ensure_layout()
    store.append_raw("- bullet one\n")
    store.append_raw("- bullet two\n")
    raw = (store.root / "raw_memories.md").read_text(encoding="utf-8")
    assert "bullet one" in raw and "bullet two" in raw
    rel = store.write_rollout("sess-1", "# Rollout\n\n内容")
    assert (store.root / rel).is_file()
    assert store.grep("内容") != "(no matches)"


def test_atomic_write_updates_memory(tmp_path) -> None:
    store = _store(tmp_path)
    store.ensure_layout()
    store.atomic_write(MEMORY_NAME, "v1\n# MEMORY\n\n新内容\n")
    assert store.memory_has_entries()
    assert "新内容" in store.read_rel(MEMORY_NAME, limit=None)


def test_clear_resets_to_templates(tmp_path) -> None:
    store = _store(tmp_path)
    store.ensure_layout()
    store.atomic_write(MEMORY_NAME, "v1\n# MEMORY\n\n内容\n")
    store.append_raw("- x\n")
    store.clear()
    assert store.summary_is_placeholder()
    assert not store.memory_has_entries()


def test_resolve_rel_rejects_escape(tmp_path) -> None:
    store = _store(tmp_path)
    store.ensure_layout()
    with pytest.raises(PermissionError):
        store.resolve_rel("../escape.md")
    with pytest.raises(PermissionError):
        store.resolve_rel("/abs/path.md")


# -- pipeline：抽取/合并 ------------------------------------------------------


def test_slice_round_collects_last_turn() -> None:
    msgs = [
        Message("user", "第一轮"),
        Message("assistant", "好的"),
        Message("user", "第二轮"),
        Message("assistant", "开始"),
        Message(
            "assistant",
            "",
            tool_calls=[ToolCall("t1", "write_file", '{"path": "a.txt"}')],
        ),
        Message("tool", "已写入 a.txt", tool_call_id="t1"),
    ]
    rc = slice_round(msgs, workspace=Path("/ws"), session_id="s1")
    assert rc.user_text == "第二轮"
    assert "write_file" in rc.tool_calls[0]
    assert "已写入" in rc.tool_outputs[0]


def test_should_extract_rules(tmp_path) -> None:
    assert should_extract(_round(tmp_path, user="")) is False
    assert should_extract(_round(tmp_path, user="长" * 10)) is True
    assert should_extract(_round(tmp_path, tool_calls=("ls()",))) is True


def test_process_batch_writes_raw_and_signal(tmp_path) -> None:
    store = _store(tmp_path)
    llm = FakeChatLLM(['{"raw_bullets": ["结论：用 Redis 缓存"], "rollout_summary": "定了缓存方案"}'])
    pipe = MemoryPipeline(store, llm, "m")
    pipe._process_batch([_round(tmp_path)])
    assert "Redis 缓存" in (store.root / "raw_memories.md").read_text(encoding="utf-8")
    assert store.grep("Redis") != "(no matches)"  # rollout 文件已落盘
    assert pipe.pending_signal_count == 1


def test_process_batch_filters_secrets(tmp_path) -> None:
    store = _store(tmp_path)
    llm = FakeChatLLM(
        ['{"raw_bullets": ["sk-abcdef0123456789abcdef 是密钥", "结论：升级"], "rollout_summary": ""}']
    )
    pipe = MemoryPipeline(store, llm, "m")
    pipe._process_batch([_round(tmp_path)])
    raw = (store.root / "raw_memories.md").read_text(encoding="utf-8")
    assert "sk-abcdef" not in raw
    assert "结论：升级" in raw


def test_stage2_triggers_at_min_signals(tmp_path) -> None:
    """攒够 CONSOLIDATE_MIN_SIGNALS 条信号 → stage2 合并 → MEMORY/summary 落盘、pending 清空。"""
    store = _store(tmp_path)
    responses = ['{"raw_bullets": ["事实1"], "rollout_summary": ""}'] * CONSOLIDATE_MIN_SIGNALS
    responses.append(STAGE2_REWRITE)
    pipe = MemoryPipeline(store, FakeChatLLM(responses), "m")
    for _ in range(CONSOLIDATE_MIN_SIGNALS):
        pipe._process_batch([_round(tmp_path, user=f"第 {_} 轮")])
    assert pipe.pending_signal_count == 0
    assert "Redis 7" in store.read_rel(MEMORY_NAME, limit=None)
    assert "Redis 7" in store.read_summary(limit=None)


def test_stage2_unchanged_consumes_pending(tmp_path) -> None:
    store = _store(tmp_path)
    responses = ['{"raw_bullets": ["事实"], "rollout_summary": ""}'] * CONSOLIDATE_MIN_SIGNALS
    responses.append('{"unchanged": true}')
    pipe = MemoryPipeline(store, FakeChatLLM(responses), "m")
    for _ in range(CONSOLIDATE_MIN_SIGNALS):
        pipe._process_batch([_round(tmp_path)])
    assert pipe.pending_signal_count == 0  # unchanged 也算消费
    assert store.summary_is_placeholder()  # 没改写文件


def test_stage2_parse_failure_keeps_pending(tmp_path) -> None:
    store = _store(tmp_path)
    responses = ['{"raw_bullets": ["事实"], "rollout_summary": ""}'] * CONSOLIDATE_MIN_SIGNALS
    responses.append("garbage output")  # stage2 解析失败
    pipe = MemoryPipeline(store, FakeChatLLM(responses), "m")
    for _ in range(CONSOLIDATE_MIN_SIGNALS):
        pipe._process_batch([_round(tmp_path)])
    assert pipe.pending_signal_count == CONSOLIDATE_MIN_SIGNALS  # 保留，等 drain 重试


def test_invalidate_discards_inflight_and_epoch(tmp_path) -> None:
    """/memory clear 的 epoch 作废：stage1 调用中作废 → 在途批次不写盘。"""
    store = _store(tmp_path)
    pipe = MemoryPipeline(store, FakeChatLLM(), "m")

    class InvalidateDuringLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            self.calls += 1
            if self.calls == 1:
                pipe.invalidate()  # stage1 调用中作废（clear 竞态）
            return SimpleNamespace(
                messages=[Message("assistant", '{"raw_bullets": ["不应落盘"], "rollout_summary": ""}')]
            )

    pipe._client = InvalidateDuringLLM()
    pipe._process_batch([_round(tmp_path)])
    raw = (store.root / "raw_memories.md").read_text(encoding="utf-8")
    assert "不应落盘" not in raw  # epoch 变了 → 在途批次丢弃


def test_submit_and_drain_threaded(tmp_path) -> None:
    """提交 → daemon worker 处理 → drain 强制合并（退出路径）。"""
    store = _store(tmp_path)
    llm = FakeChatLLM(
        ['{"raw_bullets": ["事实A"], "rollout_summary": ""}', STAGE2_REWRITE]
    )
    pipe = MemoryPipeline(store, llm, "m")
    pipe.submit(_round(tmp_path, user="线程提交"))
    pipe.drain()
    assert llm.calls  # stage1 跑过
    assert "事实A" in (store.root / "raw_memories.md").read_text(encoding="utf-8")
    assert "Redis 7" in store.read_rel(MEMORY_NAME, limit=None)  # stage2 也跑过


def test_idle_flush_after_idle_seconds(tmp_path) -> None:
    store = _store(tmp_path)
    llm = FakeChatLLM(['{"raw_bullets": ["事实"], "rollout_summary": ""}', STAGE2_REWRITE])
    pipe = MemoryPipeline(store, llm, "m", consolidate_idle_seconds=0.05)
    pipe.submit(_round(tmp_path))
    # 等空闲定时器触发 stage2（轮询，最长 2s）
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and "Redis 7" not in store.read_summary(limit=None):
        time.sleep(0.05)
    pipe.drain()
    assert "Redis 7" in store.read_summary(limit=None)


# -- agent 接线 --------------------------------------------------------------


class RecorderPipeline:
    """记录 submit 的假管线（不启线程、不调 LLM）。"""

    def __init__(self) -> None:
        self.submitted: list[RoundContent] = []

    def submit(self, rc: RoundContent) -> None:
        self.submitted.append(rc)


def test_maybe_submit_round_submits_and_noop(tmp_path) -> None:
    """有管线 → 正常回合提交切片；无管线（默认）→ 全路径 no-op。"""
    from vgent.agent import _maybe_submit_memory_round

    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    msgs = [Message("user", "优化 git 仓库"), Message("assistant", "开始")]

    recorder = RecorderPipeline()
    ctx = SessionContext(session_id=sid, store=store, llm=SimpleNamespace(), memory_pipeline=recorder)
    _maybe_submit_memory_round(ctx, msgs)
    assert len(recorder.submitted) == 1

    plain = SessionContext(session_id=sid, store=store, llm=SimpleNamespace())
    _maybe_submit_memory_round(plain, msgs)  # 无管线：no-op 不抛
    assert recorder.submitted == [recorder.submitted[0]]


def test_run_turn_submits_to_pipeline_after_turn(tmp_path) -> None:
    """run_turn 正常结束后把本轮切片提交管线；不增加 ctx.llm 调用。"""

    class Scripted:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls = []

        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            self.calls.append(list(messages))
            return self.responses.pop(0)

    def final(text):
        return ChatResult(messages=[Message("assistant", text)], usage=Usage(10, 5, 15))

    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    llm = Scripted([final("好的，开始优化")])
    recorder = RecorderPipeline()
    ctx = SessionContext(session_id=sid, store=store, llm=llm, memory_pipeline=recorder)
    run_turn("继续上次那个 git 仓库优化", ctx)
    assert len(recorder.submitted) == 1
    assert recorder.submitted[0].user_text == "继续上次那个 git 仓库优化"
    assert len(llm.calls) == 1  # 管线不增加主 LLM 调用（R3）


def test_memory_summary_injected_in_send(tmp_path) -> None:
    """ctx.memory_summary 非空 → 首调 send 含 [记忆总览]。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    recorder = RecorderPipeline()

    class Scripted:
        def __init__(self):
            self.calls = []

        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            self.calls.append(list(messages))
            return ChatResult(messages=[Message("assistant", "好")], usage=Usage(1, 1, 2))

    llm = Scripted()
    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=llm,
        memory_pipeline=recorder,
        memory_summary="Redis 7 二级缓存；TTL 300 秒。",
    )
    run_turn("继续优化", ctx)
    sent = llm.calls[0]
    assert any(m.role == "system" and "记忆总览" in m.content for m in sent)


def test_freshness_warning_on_old_entry() -> None:
    """>1 天的记忆条目注入时附新鲜度警告；新条目不加。"""
    fresh = MemoryEntry(_now_iso(), "s1", "标题", "主题", "摘要", "ws")
    old = MemoryEntry("2020-01-01T00:00:00+00:00", "s1", "标题", "主题", "摘要", "ws")
    assert "天前记忆" not in memory_note_text(fresh)
    assert "天前记忆" in memory_note_text(old)


def test_auto_recall_injects_freshness_warning(tmp_path) -> None:
    """自动回忆注入旧条目时带新鲜度警告（集成路径）。"""
    from datetime import UTC, datetime, timedelta

    from vgent.memory.episodic import current_project

    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    path = tmp_path / "m.jsonl"
    mem = EpisodicMemory(path)
    old_ts = (datetime.now(UTC) - timedelta(days=5)).isoformat(timespec="seconds")
    path.write_text(
        json.dumps(
            {
                "ts": old_ts,
                "session_id": "other",
                "title": "t",
                "topic": "缓存方案",
                "summary": "用 Redis",
                "project": current_project(),  # P5：自动回忆只搜当前项目
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Scripted:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            self.calls.append(list(messages))
            return ChatResult(messages=[Message("assistant", "好")], usage=Usage(1, 1, 2))

    llm = Scripted()
    llm.calls = []
    ctx = SessionContext(session_id=sid, store=store, llm=llm, memory=mem)
    run_turn("上次那个缓存方案", ctx)
    sent = llm.calls[0]
    hits = [m for m in sent if m.role == "system" and "记忆" in m.content]
    assert hits and "天前记忆" in hits[0].content


# -- 工具：memory_read / memory_grep -------------------------------------------


def _register_memory_tools(tmp_path) -> tuple[ToolRegistry, MemoryFileStore]:
    data_home = tmp_path / "data"
    ws = tmp_path / "ws"
    store = MemoryFileStore(data_home, ws)
    store.ensure_layout()
    store.atomic_write(MEMORY_NAME, "v1\n# MEMORY\n\n## 缓存\n- Redis 7 TTL 300（二级缓存方案）\n")
    reg = ToolRegistry()
    for t in make_memory_tools(data_home, ws):
        reg.register(t.schema, t.handler)
    return reg, store


def test_memory_read_and_grep_tools(tmp_path) -> None:
    reg, _ = _register_memory_tools(tmp_path)
    assert reg.get("memory_read") is not None
    assert reg.get("memory_grep") is not None
    out = reg.execute("memory_read", {"path": "MEMORY.md"})
    assert "Redis 7" in out
    out = reg.execute("memory_grep", {"query": "Redis 缓存"})
    assert "MEMORY.md" in out


def test_memory_read_rejects_summary_and_escape(tmp_path) -> None:
    reg, _ = _register_memory_tools(tmp_path)
    out = reg.execute("memory_read", {"path": "memory_summary.md"})
    assert "already in the system prompt" in out
    out = reg.execute("memory_read", {"path": "../secret.txt"})
    assert "error" in out
    out = reg.execute("memory_read", {})
    assert "path is required" in out


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")
