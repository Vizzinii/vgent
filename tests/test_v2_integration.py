"""全方位测试方案 v2 · C 组：跨模块集成（P0 病灶区，本方案核心）。

1. compact 全生命周期（/compact → 续聊 → 模拟新进程 → 再压缩 → boundary/retained/去重）
2. 记忆管线端到端 F4 闭环（管线写文件 ↔ agent 下一轮读到新 summary）
3. 快照 × 中断 × 恢复（write 真写盘 → KeyboardInterrupt → finally seal → /restore last）
4. F8 × Web 确认桥（Web 越界写 → SSE confirm → approve → 写入成功）
5. headless × allow × F8（allow 下 cwd 内直执行 / cwd 外无交互=拒绝）
6. 压缩 × 锚点交互（估算压缩触发后 send 重建仍含首轮锚点）
"""
from __future__ import annotations

import io
import json
import queue
import threading
import time
import urllib.error
import urllib.request

import pytest
from conftest import FakeLLM, StageClient, _chat_final, _chat_with_tools
from rich.console import Console

import vgent.cli as cli_mod
from vgent.agent import SessionContext, run_turn
from vgent.cli import _compact_inline, _dispatch_command
from vgent.config import Config, ContextConfig, PermissionRules
from vgent.context import ContextEngine
from vgent.memory.pipeline import MemoryPipeline
from vgent.memory.store import MemoryFileStore
from vgent.messages import Message, ToolCall
from vgent.permission import ConfirmResult, PermissionSystem
from vgent.snapshot import SnapshotStore
from vgent.store import SessionStore
from vgent.tools import default_tools
from vgent.web.server import HubManager, make_server

# ===========================================================================
# C1 compact 全生命周期
# ===========================================================================


def test_compact_full_lifecycle(tmp_path) -> None:
    """C1：/compact（落库）→ 续聊 3 轮 send 必含前轮 → 新进程再 /compact →
    boundary 单调递增、retained 与边界后消息无重叠、发送列表无重复。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    # 窗口取大（V2 后自动压缩会真发生）：本测聚焦手动 /compact 生命周期，
    # 避免续聊轮触发 estimated 路径覆盖压缩记录（该路径由 C6 专门覆盖）
    engine = ContextEngine(context_length=50_000, cfg=ContextConfig(tail_token_budget=60))
    llm = FakeLLM()
    ctx = SessionContext(session_id=sid, store=store, llm=llm, engine=engine)
    console = Console(file=io.StringIO(), force_terminal=False)

    # 预置 6 条历史（内容较长，保证中间段可压缩）
    for i in range(3):
        store.add_messages(
            sid,
            [
                Message("user", f"问题{i}：" + "长" * 400),
                Message("assistant", f"回答{i}：" + "答" * 400),
            ],
        )

    _compact_inline(ctx, console)  # 第一次 /compact
    c1 = store.get_compact(sid)
    assert c1 is not None
    summary1, retained1, boundary1 = c1

    # 续聊 3 轮：每轮 send 必含前轮消息（F1 修复语义：不丢中间轮）
    for i in range(3):
        run_turn(f"续聊第{i}轮", ctx)
    for i in range(1, 3):
        prev_text = f"续聊第{i - 1}轮"
        assert any(getattr(m, "content", "") == prev_text for m in llm.calls[i]), (
            f"第 {i} 轮 send 缺第 {i - 1} 轮用户消息（丢上下文）"
        )

    # 模拟新进程：新 SessionContext（同 store 同 sid，新 engine/llm）
    engine2 = ContextEngine(context_length=50_000, cfg=ContextConfig(tail_token_budget=60))
    llm2 = FakeLLM()
    ctx2 = SessionContext(session_id=sid, store=store, llm=llm2, engine=engine2)
    run_turn("新进程续聊", ctx2)
    send = llm2.calls[0]
    # 新进程 send = 压缩底稿（头+摘要+尾部+边界后）+ 当前消息，而非全量历史
    assert any(m.role == "system" and summary1[:20] in m.content for m in send)
    # 被压缩的中间段不出现（头部首条「问题0」按设计保留，检查中间段消息）
    assert not any(getattr(m, "content", "").startswith("回答0") for m in send)
    assert not any(getattr(m, "content", "").startswith("问题1") for m in send)
    # 发送列表无重复（按 user 内容断言唯一）
    user_texts = [m.content for m in send if m.role == "user"]
    assert len(user_texts) == len(set(user_texts)), "发送列表出现重复消息"

    _compact_inline(ctx2, console)  # 新进程里再 /compact
    c2 = store.get_compact(sid)
    assert c2 is not None
    _, retained2, boundary2 = c2
    assert boundary2 > boundary1, "两次压缩边界应单调递增"
    # retained 与边界之后的消息无重叠（retained 是边界前真相的尾部切片）
    after = store.get_history_after(sid, boundary2)
    retained_texts = {m.content for m in retained2}
    overlap = retained_texts & {m.content for m in after}
    assert not overlap, f"retained 与边界后消息重叠：{overlap}"
    assert retained1 is not None
    store.close()


# ===========================================================================
# C2 记忆管线端到端（F4 闭环）
# ===========================================================================


def test_memory_pipeline_end_to_end_summary_refresh(tmp_path, monkeypatch) -> None:
    """C2：memory_auto 管线 3 轮触发 stage2 写新 summary → 下一轮 send 的
    [记忆总览] 是新内容（管线写文件 ↔ agent 读文件的闭环，F4）。"""
    monkeypatch.chdir(tmp_path)
    store_db = SessionStore(tmp_path / "t.db")
    sid = store_db.create_session()
    mem_store = MemoryFileStore(tmp_path, tmp_path)
    client = StageClient(stage2_mode="ok")
    pipe = MemoryPipeline(mem_store, client, "m", consolidate_min_signals=3, consolidate_idle_seconds=9999)
    llm = FakeLLM()
    ctx = SessionContext(
        session_id=sid,
        store=store_db,
        llm=llm,
        memory_file_store=mem_store,
        memory_pipeline=pipe,
    )

    # 连跑 3 轮（每轮文本足够长触发 should_extract）→ 攒满 3 信号触发 stage2
    for i in range(3):
        run_turn(f"第{i}轮：我们决定用 Redis 做缓存 TTL 三百秒，这是重要决策", ctx)

    deadline = time.time() + 10
    while time.time() < deadline:
        if "新总览" in mem_store.read_summary():
            break
        time.sleep(0.1)
    assert "新总览" in mem_store.read_summary(), "stage2 未写出新 summary"

    # 下一轮 send 的 [记忆总览] 必须是刚写的新 summary（F4：每轮重读文件）
    run_turn("第四轮", ctx)
    last_send = llm.calls[-1]
    overview = [m for m in last_send if m.role == "system" and m.content.startswith("[记忆总览]")]
    assert overview, "send 缺 [记忆总览] 注入"
    assert "新总览" in overview[0].content
    pipe.drain()
    store_db.close()


# ===========================================================================
# C3 快照 × 中断 × 恢复
# ===========================================================================


class _InterruptLLM:
    """第一次 chat 返回 write_file 工具调用；第二次（工具执行后续轮）抛 KeyboardInterrupt。"""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls += 1
        if self.calls == 1:
            return _chat_with_tools(
                ToolCall("c1", "write_file", json.dumps({"path": "f.txt", "content": "NEW"}))
            )
        raise KeyboardInterrupt("用户中断")


def test_snapshot_interrupt_and_restore_last(tmp_path, monkeypatch) -> None:
    """C3：write_file 真写盘 → run_turn 中途 KeyboardInterrupt → finally seal →
    /restore last 恢复写前内容（CLI dispatch 层集成）。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    f = ws / "f.txt"
    f.write_text("OLD", encoding="utf-8")

    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    snaps = SnapshotStore(tmp_path / "ck", ws)
    llm = _InterruptLLM()
    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=llm,
        tools=default_tools(),
        permissions=PermissionSystem(confirm=lambda t, a: ConfirmResult.APPROVE),
        snapshots=snaps,
    )

    with pytest.raises(KeyboardInterrupt):
        run_turn("改文件", ctx)
    assert f.read_text(encoding="utf-8") == "NEW"  # 工具已执行写盘
    assert snaps.snapshot_count() == 1  # finally seal_turn 已封存（含写前登记）

    # CLI dispatch 层 /restore last → 文件回到写前内容
    console = Console(file=io.StringIO(), force_terminal=False)
    ok = _dispatch_command(
        "/restore last", ctx, console, lambda ps: "", tmp_path / "last", {"n": 0}
    )
    assert ok is True
    assert f.read_text(encoding="utf-8") == "OLD"
    assert "已恢复" in console.file.getvalue()
    store.close()


# ===========================================================================
# C4 F8 × Web 确认桥
# ===========================================================================


class _SSECollector:
    """后台线程收集某会话的 SSE 事件，按名等待。"""

    def __init__(self, url: str) -> None:
        self.events: list[tuple[str | None, dict]] = []
        self.q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.url = url

    def _run(self) -> None:
        try:
            with urllib.request.urlopen(self.url, timeout=10) as r:
                buf = b""
                while not self.stop.is_set():
                    chunk = r.read(1)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n\n" in buf:
                        frame, buf = buf.split(b"\n\n", 1)
                        ev = None
                        data: dict = {}
                        for line in frame.decode("utf-8", errors="replace").split("\n"):
                            if line.startswith("event: "):
                                ev = line[7:]
                            elif line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                except json.JSONDecodeError:
                                    pass
                        self.q.put((ev, data))
        except OSError:
            pass  # 测试收尾 server 先关导致的 socket 竞态（关键事件已在此之前收到）

    def wait(self, name: str, timeout: float = 10.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                ev, data = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            self.events.append((ev, data))
            if ev == name:
                return data
        raise AssertionError(f"SSE 事件未到达：{name}")

    def start(self) -> _SSECollector:
        self.thread.start()
        return self

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def test_web_f8_outside_write_confirms_via_sse(tmp_path, monkeypatch) -> None:
    """C4：Web 端模型 write_file 到 cwd 外 → SSE confirm 事件到达（非静默执行）→
    回传 approve → 文件写入成功（越界=确认而非拒绝的语义验证）。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    target = tmp_path / "outside.txt"  # cwd 外

    class WriteOutsideLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            self.calls += 1
            if self.calls == 1:
                return _chat_with_tools(
                    ToolCall("c1", "write_file", json.dumps({"path": str(target), "content": "OK"}))
                )
            return _chat_final("写好了")

    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    manager = HubManager(cfg, store, WriteOutsideLLM(), default_tools())
    sid = store.create_session()
    httpd = make_server(manager, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    sse = _SSECollector(f"{base}/api/sessions/{sid}/stream").start()
    result_box: dict = {}

    def post_message():
        req = urllib.request.Request(
            f"{base}/api/sessions/{sid}/messages",
            data=json.dumps({"content": "写文件到外面"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                result_box["code"] = r.status
                result_box["body"] = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            result_box["code"] = e.code
            result_box["body"] = json.loads(e.read().decode("utf-8"))

    try:
        t = threading.Thread(target=post_message)
        t.start()
        confirm_data = sse.wait("confirm")  # 必须先弹确认（F8），而非静默写
        assert confirm_data["tool"] == "write_file"
        req = urllib.request.Request(
            f"{base}/api/sessions/{sid}/confirm",
            data=json.dumps({"result": "approve"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 200
        t.join(timeout=30)
        assert not t.is_alive()
        assert result_box.get("code") == 200
        assert target.read_text(encoding="utf-8") == "OK"  # approve 后写入成功
        sse.wait("done")
    finally:
        sse.close()
        httpd.shutdown()
        httpd.server_close()
        store.close()


# ===========================================================================
# C5 headless × allow × F8
# ===========================================================================


def _run_headless_with(llm, tmp_path, monkeypatch, rules: PermissionRules) -> tuple[int, list]:
    """隔离环境下跑 _run_headless：cwd=tmp_path/ws，LLMClient 注入桩。"""
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    monkeypatch.chdir(ws)
    monkeypatch.setattr(cli_mod, "LLMClient", lambda cfg: llm)
    cfg = Config(data_dir=tmp_path, permissions=rules)
    store = SessionStore(tmp_path / "t.db")
    console = Console(file=io.StringIO(), force_terminal=False)
    code = cli_mod._run_headless(cfg, store, "写文件", console)
    sessions = store.list_sessions()
    store.close()
    return code, sessions


def test_headless_allow_inside_executes(tmp_path, monkeypatch) -> None:
    """C5a：allow=["write_file"] 下 headless 写 cwd 内 → 规则放行直接执行。"""
    inside = tmp_path / "ws" / "inside.txt"

    class WriteLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            self.calls += 1
            if self.calls == 1:
                return _chat_with_tools(
                    ToolCall("c1", "write_file", json.dumps({"path": "inside.txt", "content": "IN"}))
                )
            return _chat_final("done")

    code, sessions = _run_headless_with(
        WriteLLM(), tmp_path, monkeypatch, PermissionRules(allow=["write_file"])
    )
    assert code == 0
    assert sessions == []  # F12：headless 会话清理
    assert inside.read_text(encoding="utf-8") == "IN"


def test_headless_allow_outside_rejected(tmp_path, monkeypatch) -> None:
    """C5b：allow=["write_file"] 下 headless 写 cwd 外 → F8 强制确认 →
    无交互（confirm=None）= 拒绝 → 文件不写、工具拒绝消息回喂。"""
    target = tmp_path / "outside.txt"
    seen: list[list[Message]] = []

    class WriteLLM:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            self.calls += 1
            seen.append(list(messages))
            if self.calls == 1:
                return _chat_with_tools(
                    ToolCall("c1", "write_file", json.dumps({"path": str(target), "content": "OUT"}))
                )
            return _chat_final("done")

    code, _ = _run_headless_with(
        WriteLLM(), tmp_path, monkeypatch, PermissionRules(allow=["write_file"])
    )
    assert code == 0
    assert not target.exists()  # 越界 + 无交互 → 未写
    assert any(
        m.role == "tool" and "拒绝了工具" in m.content for m in seen[1]
    ), "第二次 chat 的 send 应含拒绝回喂"


# ===========================================================================
# C6 压缩 × 锚点交互
# ===========================================================================


def test_compress_then_rebuild_keeps_anchors(tmp_path) -> None:
    """C6：首轮注入 instructions/cwd 锚点 → 估算压缩触发 → 重建后的 send
    仍含锚点（锚点消费时序与 compress-then-rebuild 组合）。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    # reserved=800 > watermark(750)：任何 send 都触发估算压缩路径
    engine = ContextEngine(
        context_length=1000,
        cfg=ContextConfig(tail_token_budget=50, reserved_output_tokens=800),
    )
    llm = FakeLLM()
    ctx = SessionContext(
        session_id=sid,
        store=store,
        llm=llm,
        engine=engine,
        instructions="项目指令XYZ",
        instructions_name="AGENTS.md",
    )
    # 预置压缩底稿（模拟上轮 /compact 后的引擎态）：store 无历史 → first_turn=True
    # 锚点照常注入，同一次循环迭代内「组装 send → 估算压缩 → 重建 send」
    engine.compacted = [
        Message("system", "头"),
        Message("user", "u" * 1200),
        Message("assistant", "a" * 1200),
    ]
    run_turn("新问题", ctx)
    send = llm.calls[0]
    has_anchor = any(
        m.role == "system" and "项目指令XYZ" in m.content and "AGENTS.md" in m.content for m in send
    )
    has_cwd = any(m.role == "system" and m.content.startswith("工作目录：") for m in send)
    has_marker = any(m.role == "system" and "已由 TailWindow 压缩" in m.content for m in send)
    assert has_marker, "估算压缩未触发（测试环境不对）"
    assert has_anchor, "压缩重建后的 send 丢了 instructions 锚点"
    assert has_cwd, "压缩重建后的 send 丢了 cwd 锚点"
    # 锚点在压缩标记之前（引导信息优先于历史）
    idx_anchor = next(i for i, m in enumerate(send) if m.role == "system" and "项目指令XYZ" in m.content)
    idx_marker = next(i for i, m in enumerate(send) if "已由 TailWindow 压缩" in m.content)
    assert idx_anchor < idx_marker
    store.close()


def test_estimated_trigger_compresses_even_if_heuristic_below(tmp_path) -> None:
    """C6 补充：估算路径触发（schema 开销顶过阈值）而启发式口径低于水位 →
    压缩应当实际发生（V2 修复：estimated 分支 force=True）。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    engine = ContextEngine(
        context_length=1000,
        cfg=ContextConfig(tail_token_budget=50, reserved_output_tokens=800),
    )
    llm = FakeLLM()
    ctx = SessionContext(session_id=sid, store=store, llm=llm, engine=engine, tools=default_tools())
    # 历史本体启发式 ~420 < watermark(750)，但 send + tools schema(tiktoken ~800+) + 800 预留 → 估算触发
    store.add_messages(
        sid,
        [
            Message("system", "头"),
            Message("user", "u" * 600),
            Message("assistant", "a" * 600),
        ],
    )
    assert engine.should_compress_estimated(
        [*store.get_history(sid), Message("user", "新问题")],
        model=None,
        fixed_extra=json.dumps(default_tools().schemas(), ensure_ascii=False),
    ), "前置校验：估算口径确实触发"
    run_turn("新问题", ctx)
    send = llm.calls[0]
    assert any(m.role == "system" and "已由 TailWindow 压缩" in m.content for m in send), (
        "估算触发但压缩未发生"
    )
    store.close()
