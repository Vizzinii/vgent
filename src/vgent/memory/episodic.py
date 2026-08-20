"""v2 演进 · episodic 记忆（M8）：跨会话任务摘要（做了什么/结论/遗留）。

- 存储：本机 `~/.vgent/memory/episodic.jsonl`（append-only UTF-8，不进同步盘，决策 7）；
- 生成：LLM 把最近会话压缩成 3~5 句摘要（best-effort，异常/空响应返回 ""，不阻断对话）；
- 检索：关键词子串匹配（v1 不做向量），命中注入上下文；
- 触发：`/remember <主题>` 显式存储；`/recall <关键词>` 显式检索并落库注入；
  `/memories` 列出；`memory_auto=true` 时任务计划完成自动存储（每会话去重）。
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vgent.messages import Message

SUMMARIZE_PROMPT = (
    "你是会话记忆整理器。把下面的对话压缩成任务摘要，要求：\n"
    "1. 用要点句输出 3~5 句，覆盖：做了什么、关键结论、遗留事项；\n"
    "2. 用自己的话概括，绝对不要引用或复述对话中的原句；\n"
    "3. 只输出摘要本身，不要任何前缀或解释。"
)

_TAIL_WINDOW = 30  # 摘要只看最近 N 条（够用，省 token）
# 摘要最短长度：短于此判定为失败（对话碎片/复读原句），不写入记忆
_MIN_SUMMARY_CHARS = 20


@dataclass
class MemoryEntry:
    ts: str
    session_id: str
    title: str
    topic: str
    summary: str

    def to_line(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)

    @classmethod
    def from_line(cls, line: str) -> MemoryEntry | None:
        try:
            data = json.loads(line)
            return cls(
                str(data["ts"]),
                str(data["session_id"]),
                str(data["title"]),
                str(data["topic"]),
                str(data["summary"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None  # 坏行跳过，不阻断


class EpisodicMemory:
    def __init__(self, path: Path) -> None:
        self.path = path

    def add(self, topic: str, summary: str, session_id: str, title: str) -> MemoryEntry:
        entry = MemoryEntry(_now(), session_id, title, topic, summary)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(entry.to_line() + "\n")
        return entry

    def has_session(self, session_id: str) -> bool:
        """该会话是否已存过摘要（自动存储去重用）。"""
        return any(e.session_id == session_id for e in self._entries())

    def search(self, keyword: str, limit: int = 3) -> list[MemoryEntry]:
        """关键词匹配（大小写不敏感，双向）：keyword 命中 topic/summary，
        或 topic 出现在 keyword 里（自动回忆：用户消息提到上次主题）；返回最近 limit 条。"""
        kw = keyword.strip().lower()
        if not kw:
            return []
        hits = [
            e for e in self._entries()
            if kw in e.topic.lower() or kw in e.summary.lower() or e.topic.lower() in kw
        ]
        return hits[-limit:]

    def list_recent(self, limit: int = 10) -> list[MemoryEntry]:
        return self._entries()[-limit:]

    def count(self) -> int:
        return len(self._entries())

    def _entries(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        return [e for line in lines if (e := MemoryEntry.from_line(line)) is not None]


def summarize(msgs: list[Message], llm: Callable, topic: str) -> str:
    """LLM 压缩最近历史为任务摘要；异常/空响应/质量不过关返回 ""（best-effort，不阻断）。

    deepseek 思考模式下模型可能把摘要放思考流、正文只吐一句碎片（真机首跑踩坑：
    /remember 存进「好的，我再试一次。」）——正文过短时退回 reasoning_content，
    仍过短则判失败不写入。
    """
    prompt = Message("system", SUMMARIZE_PROMPT)
    try:
        result = llm.chat([prompt, *msgs[-_TAIL_WINDOW:]])
    except Exception:  # noqa: BLE001 — 摘要失败静默，不阻断对话
        return ""
    msg = result.messages[0]
    text = (msg.content or "").strip()
    if len(text) < _MIN_SUMMARY_CHARS:
        text = (msg.reasoning_content or "").strip()
    if len(text) < _MIN_SUMMARY_CHARS:
        return ""
    return text


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
