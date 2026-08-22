"""M12-C 长期记忆写入管线：对话轮次 → stage1 抽取 → 防抖 stage2 合并。

思路来源（只学思路不抄代码）：xcode-py memory/pipeline.py 的 async 管线，
vgent 适配为 **sync**（queue + daemon 线程 + threading.Timer 替代 asyncio）。

端到端时序（run_turn 正常结束后）：
1. agent 调 submit(slice_round(session.messages)) —— 非阻塞入队，不调 ctx.llm（R3）
2. 后台 worker（daemon 线程）取批（尽量掏空队列合并多轮）：
   a. stage1 LLM → bullets + rollout_summary
   b. 密钥黑名单过滤（sk-xxx / api_key= 等）
   c. 写 rollout 文件（可选）+ append raw_memories
   d. 拼一条 signal 进 pending
   e. pending ≥ 3 → 立刻 stage2；否则 arm 300s 空闲定时器
3. stage2：读**完整** MEMORY + summary + 本批 signals → LLM → 原子写两文件
   - 成功或 unchanged 才从 pending 消费对应 signals
   - 失败/解析失败：**保留 pending**，等 drain 或下次再试
4. 进程退出（cli finally / web serve finally）：drain() 有界重试强制 flush

并发与 clear 安全：
- 单 worker 串行消费队列；stage2 用独立锁保证同一时刻只有一次合并
- _epoch：/memory clear 时 +1 并清空 queue/pending；在途 stage1/2 看到 epoch
  变了就丢弃写盘（防 clear 竞态）

与会话历史：stage1 吃的是**送模侧** messages（tool 可能已 prune），不是 SQLite 全文。
这是有意的：记忆只要稳定事实，不需要巨型 tool 原文。
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vgent.memory.prompts import (
    STAGE1_SYSTEM,
    STAGE2_SYSTEM,
    is_blacklisted,
    parse_stage1,
    parse_stage2,
)
from vgent.memory.store import (
    MEMORY_NAME,
    SUMMARY_NAME,
    MemoryFileStore,
    format_raw_append,
)

logger = logging.getLogger(__name__)

TOOL_OUTPUT_PER_ITEM = 200  # stage1 输入里单条 tool 输出片段上限
TOOL_OUTPUT_TOTAL = 2000  # stage1 输入里 tool 输出总预算
CONSOLIDATE_MIN_SIGNALS = 3  # 攒够 N 条信号触发 stage2
CONSOLIDATE_IDLE_SECONDS = 300.0  # 空闲防抖：距上条信号这么久触发 stage2
DRAIN_MAX_ROUNDS = 8  # 退出时最多重试轮数
DRAIN_JOIN_TIMEOUT = 5.0  # 每轮等 worker 的超时（秒）
_TRIVIAL_USER = 8  # 用户话短于此不抽取
_TRIVIAL_TOTAL = 16  # 用户+助手总文本短于此不抽取


@dataclass(frozen=True, slots=True)
class RoundContent:
    """一轮对话的不可变快照（stage1 输入）。"""

    workspace: str
    session_id: str
    user_text: str
    assistant_texts: tuple[str, ...] = ()
    tool_calls: tuple[str, ...] = ()
    tool_outputs: tuple[str, ...] = ()


def slice_round(
    messages: list[Any],
    *,
    workspace: Path,
    session_id: str,
) -> RoundContent:
    """从完整 messages 切出「最近一轮用户回合」供 stage1。

    从最后一条 user 起，收集该 user、后续 assistant 文本、tool_calls 摘要、
    tool 输出片段（有总预算，避免 stage1 提示本身过长）。
    """
    last_user = -1
    for index, message in enumerate(messages):
        if getattr(message, "role", None) == "user":
            last_user = index
    last_user = max(last_user, 0)

    user_text = ""
    assistant_texts: list[str] = []
    tool_calls: list[str] = []
    tool_outputs: list[str] = []
    out_budget = TOOL_OUTPUT_TOTAL

    for message in messages[last_user:]:
        role = getattr(message, "role", None)
        if role == "user":
            user_text = str(getattr(message, "content", None) or "")
        elif role == "assistant":
            text = str(getattr(message, "content", None) or "")
            if text:
                assistant_texts.append(text)
            for tc in getattr(message, "tool_calls", None) or []:
                name = str(getattr(tc, "name", "?"))
                arguments = str(getattr(tc, "arguments", "") or "")
                if len(arguments) > 80:
                    arguments = arguments[:79] + "…"
                tool_calls.append(f"{name}({arguments})")
        elif role == "tool":
            snippet = str(getattr(message, "content", None) or "").strip()[:TOOL_OUTPUT_PER_ITEM]
            if snippet and out_budget > 0:
                tool_outputs.append(snippet)
                out_budget -= len(snippet)

    return RoundContent(
        workspace=str(Path(workspace).resolve()),
        session_id=session_id,
        user_text=user_text,
        assistant_texts=tuple(assistant_texts),
        tool_calls=tuple(tool_calls),
        tool_outputs=tuple(tool_outputs),
    )


def should_extract(round_content: RoundContent) -> bool:
    """是否值得跑 stage1：有工具调用、或用户话够长、或总文本够长。"""
    if round_content.tool_calls:
        return True
    text_len = len(round_content.user_text) + sum(
        len(text) for text in round_content.assistant_texts
    )
    if len(round_content.user_text.strip()) >= _TRIVIAL_USER:
        return True
    return text_len >= _TRIVIAL_TOTAL


def _combine_rounds(batch: list[RoundContent]) -> str:
    parts: list[str] = []
    for index, rc in enumerate(batch, start=1):
        parts.append(f"--- 轮次 {index} ---")
        parts.append(f"用户：{rc.user_text}")
        if rc.assistant_texts:
            parts.append("助手：" + "\n".join(rc.assistant_texts))
        if rc.tool_calls:
            parts.append("工具调用：" + "；".join(rc.tool_calls))
        if rc.tool_outputs:
            parts.append("工具输出（截断）：" + "；".join(rc.tool_outputs))
    return "\n".join(parts)


def _format_new_signal(
    bullets: list[str], rollout_summary: str, rollout_rel: str | None
) -> str:
    parts: list[str] = []
    if bullets:
        parts.append("bullets:\n" + "\n".join(f"- {b}" for b in bullets))
    if rollout_summary:
        parts.append("rollout_summary:\n" + rollout_summary)
    if rollout_rel:
        parts.append(f"rollout_path: {rollout_rel}")
    return "\n\n".join(parts) if parts else "(none)"


class MemoryPipeline:
    """单个 workspace 的写入状态机：队列 + 单 worker + pending 信号 + 空闲定时器。

    client：duck-typed（LLMClient 或测试 fake），`chat(messages) -> ChatResult`；
    model：stage1/2 用（cli 注入 light_model 或主模型）。
    """

    def __init__(
        self,
        store: MemoryFileStore,
        client: Any,
        model: str,
        *,
        consolidate_min_signals: int = CONSOLIDATE_MIN_SIGNALS,
        consolidate_idle_seconds: float = CONSOLIDATE_IDLE_SECONDS,
    ) -> None:
        self._store = store
        self._client = client
        self._model = model
        self._consolidate_min_signals = consolidate_min_signals
        self._consolidate_idle_seconds = consolidate_idle_seconds
        self._queue: queue.Queue[RoundContent] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._idle_task: threading.Timer | None = None
        self._pending_signals: list[str] = []
        self._epoch = 0
        self._lock = threading.Lock()  # 保护 pending/idle/epoch
        self._flush_lock = threading.Lock()  # 同一时刻只跑一次 stage2
        store.ensure_layout()

    # -- 状态（展示用） ------------------------------------------------------

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def pending_signal_count(self) -> int:
        with self._lock:
            return len(self._pending_signals)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._worker is not None and self._worker.is_alive()

    # -- 入口 ---------------------------------------------------------------

    def submit(self, item: RoundContent) -> None:
        """非阻塞入队；若 worker 没在跑则拉起。不在本方法里等 LLM。"""
        self._queue.put_nowait(item)
        self._ensure_worker()

    def invalidate(self) -> None:
        """作废在途任务：清空 queue/pending，bump epoch（/memory clear 用）。"""
        with self._lock:
            self._epoch += 1
            self._pending_signals.clear()
            if self._idle_task is not None:
                self._idle_task.cancel()
                self._idle_task = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def drain(self) -> None:
        """进程退出前尽量把 queue 与 pending 刷完（有界重试）。

        失败保留 pending（raw 已在磁盘），不会无限重试。
        """
        for _ in range(DRAIN_MAX_ROUNDS):
            with self._lock:
                worker = self._worker
                idle = self._idle_task
                if idle is not None:
                    idle.cancel()
                    self._idle_task = None
            if worker is not None and worker.is_alive():
                worker.join(timeout=DRAIN_JOIN_TIMEOUT)
            try:
                self._flush_consolidate()
            except Exception:  # noqa: BLE001 — 失败保留 pending，下一轮再试
                logger.warning("memory drain consolidate failed for %s", self._store.root)
                continue
            if self._queue.empty() and self.pending_signal_count == 0:
                return
            if not self._queue.empty():
                self._ensure_worker()

    # -- worker -------------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._worker_loop, name="vgent-memory", daemon=True
                )
                self._worker.start()

    def _worker_loop(self) -> None:
        """串行消费：每次尽量把队列掏成一批，减少 stage1 调用次数。"""
        while True:
            try:
                batch = [self._queue.get(timeout=0.5)]
            except queue.Empty:
                return  # 队列空 → worker 退出；下次 submit 再拉起
            while not self._queue.empty():
                batch.append(self._queue.get_nowait())
            try:
                self._process_batch(batch)
            except Exception:
                # 评审 F14：失败批次丢弃不重排——_process_batch 可能已写 raw/rollout，
                # 重排会重复写；记忆本就 best-effort，丢一轮可接受
                logger.warning(
                    "memory pipeline batch failed for %s; dropping %d rounds",
                    self._store.root,
                    len(batch),
                    exc_info=True,
                )
                return

    def _process_batch(self, batch: list[RoundContent]) -> None:
        """一批轮次的 stage1 + 可能触发的 stage2；全程用 epoch 防 clear 竞态。"""
        epoch = self._epoch
        round_text = _combine_rounds(batch)
        stage1_user = f"## 本轮对话\n{round_text}"
        stage1_text = self._chat_text(STAGE1_SYSTEM, stage1_user)
        if epoch != self._epoch:
            return

        bullets, rollout_summary = parse_stage1(stage1_text)
        bullets = [b for b in bullets if not is_blacklisted(b)]
        if is_blacklisted(rollout_summary):
            rollout_summary = ""
        if not bullets and not rollout_summary:
            return

        session_id = batch[-1].session_id
        rollout_rel: str | None = None
        if rollout_summary:
            body = f"# Rollout summary\n\nsession: {session_id}\n\n{rollout_summary}\n"
            if bullets:
                body += "\n## bullets\n" + "\n".join(f"- {b}" for b in bullets) + "\n"
            if epoch != self._epoch:
                return
            rollout_rel = self._store.write_rollout(session_id, body)

        if epoch != self._epoch:
            return
        self._store.append_raw(
            format_raw_append(session_id=session_id, bullets=bullets, rollout_rel=rollout_rel)
        )

        signal = _format_new_signal(bullets, rollout_summary, rollout_rel)
        with self._lock:
            if epoch != self._epoch:
                return
            self._pending_signals.append(signal)
            signal_count = len(self._pending_signals)
        if signal_count >= self._consolidate_min_signals:
            try:
                self._flush_consolidate()
            except Exception:
                logger.warning("memory consolidate failed for %s", self._store.root, exc_info=True)
        else:
            self._arm_idle_flush()

    # -- stage2 -------------------------------------------------------------

    def _arm_idle_flush(self) -> None:
        with self._lock:
            if self._idle_task is not None:
                self._idle_task.cancel()
            self._idle_task = threading.Timer(
                self._consolidate_idle_seconds, self._idle_flush
            )
            self._idle_task.daemon = True
            self._idle_task.start()

    def _idle_flush(self) -> None:
        """定时器线程回调：有 pending 就合并一次。"""
        if self.pending_signal_count == 0:
            return
        try:
            self._flush_consolidate()
        except Exception:  # noqa: BLE001
            logger.warning("memory idle consolidate failed for %s", self._store.root)

    def _flush_consolidate(self) -> None:
        """成功写盘或 unchanged 后才消费 pending；失败保留 pending。

        全程在 _flush_lock 下：worker 与定时器线程不会同时合并。
        """
        with self._flush_lock:
            with self._lock:
                if not self._pending_signals:
                    return
                if self._idle_task is not None:
                    self._idle_task.cancel()
                    self._idle_task = None
                epoch = self._epoch
                signals = list(self._pending_signals)
            new_signal = "\n\n---\n\n".join(signals)

            self._store.ensure_layout()
            try:
                memory_md = self._store.read_rel(MEMORY_NAME, limit=None)
            except FileNotFoundError:
                memory_md = ""
            summary_md = self._store.read_summary(limit=None)

            user = (
                f"## 现有 MEMORY.md（完整，禁止丢弃未见部分）\n{memory_md}\n\n"
                f"## 现有 memory_summary.md（完整）\n{summary_md}\n\n"
                f"## 本批新信号（可含多轮）\n{new_signal}\n"
            )
            text = self._chat_text(STAGE2_SYSTEM, user)

            with self._lock:
                if epoch != self._epoch:
                    return
            unchanged, new_memory, new_summary = parse_stage2(text)
            if unchanged:
                self._drop_pending_prefix(signals)
                return
            if not new_memory or not new_summary:
                logger.warning(
                    "memory consolidate parse failed for %s; keeping pending", self._store.root
                )
                return

            with self._lock:
                if epoch != self._epoch:
                    return
            self._store.atomic_write(MEMORY_NAME, new_memory)
            self._store.atomic_write(SUMMARY_NAME, new_summary)
            self._drop_pending_prefix(signals)

    def _drop_pending_prefix(self, signals: list[str]) -> None:
        """消费已成功处理的信号；并发追加的新信号保留。"""
        with self._lock:
            if self._pending_signals[: len(signals)] == signals:
                del self._pending_signals[: len(signals)]
                return
            for signal in signals:
                try:
                    self._pending_signals.remove(signal)
                except ValueError:
                    pass

    def _chat_text(self, system: str, user: str) -> str:
        """调管线自己的 LLM client（绝不用 ctx.llm，R3）；异常返回空串。"""
        from vgent.messages import Message

        try:
            result = self._client.chat([Message("system", system), Message("user", user)])
        except Exception:  # noqa: BLE001 — 抽取失败静默，不阻断对话
            return ""
        msg = result.messages[0] if result.messages else None
        if msg is None:
            return ""
        text = (msg.content or "").strip()
        if not text and getattr(msg, "reasoning_content", None):
            # deepseek 思考模式：正文空时退回思考流（与 episodic.summarize 同口径）
            text = (msg.reasoning_content or "").strip()
        return text


def make_pipeline_for_workspace(
    data_home: Path,
    workspace: Path,
    client: Any,
    model: str,
) -> MemoryPipeline:
    """按 workspace 建管线（cli/web 接线用；model 已解析 light_model or 主模型）。"""
    store = MemoryFileStore(data_home, workspace)
    return MemoryPipeline(store, client, model)
