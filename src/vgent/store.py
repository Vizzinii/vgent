"""④ 会话存储：SQLite（双表 + thread_id，WAL）。

蓝本：openai-agents SQLiteSession（agent_sessions + agent_messages 双表、WAL）；
thread_id 主键概念取自 langgraph。路径在本机 ~/.vgent/sessions/，不进同步盘（决策 7）。
M11：Web UI 多线程访问——check_same_thread=False + RLock 串行化 + busy_timeout。
"""
from __future__ import annotations

import functools
import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vgent.messages import Message, ToolCall


@dataclass
class SessionMeta:
    id: str
    title: str
    created_at: str
    message_count: int


def _locked(method):
    """串行化 store 访问：Web UI 多线程与 CLI 单线程都安全。"""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class SessionStore:
    """vgent 会话库。

    两张表：
      sessions(id TEXT PRIMARY KEY, title, created_at)
      messages(id AUTOINCREMENT, session_id, role, content, tool_calls, tool_call_id, ts)
    全量历史留库（决策 8 ⑤：压缩只影响发送列表，真相永远在库里）。
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "session_id TEXT NOT NULL REFERENCES sessions(id),"
            "role TEXT NOT NULL, content TEXT NOT NULL,"
            "reasoning_content TEXT, tool_calls TEXT, tool_call_id TEXT, ts TEXT NOT NULL)"
        )
        # 旧库迁移：真实冒烟前建过的库没有 reasoning_content 列，补上
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(messages)")}
        if "reasoning_content" not in cols:
            self._conn.execute("ALTER TABLE messages ADD COLUMN reasoning_content TEXT")
        # M6：Agent 状态（每轮结束写当前状态，供恢复/展示）
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS session_states ("
            "session_id TEXT PRIMARY KEY REFERENCES sessions(id),"
            "state TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        # M12：压缩记录（摘要 + 保留尾部 + 边界）——恢复会话后重建发送底稿，
        # 不发全量历史；messages 表仍是全量真相，本表只是压缩视图的落盘。
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS session_compacts ("
            "session_id TEXT PRIMARY KEY REFERENCES sessions(id),"
            "summary TEXT NOT NULL, retained TEXT NOT NULL,"
            "boundary_id INTEGER NOT NULL, updated_at TEXT NOT NULL)"
        )
        # 评审 F16：messages 按 session 查询加索引（get_history 原为全表扫描）
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
        )
        self._conn.commit()

    @_locked
    def close(self) -> None:
        self._conn.close()

    # -- 会话 CRUD ---------------------------------------------------------

    @_locked
    def create_session(self, title: str = "新会话") -> str:
        sid = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
            (sid, title, _now()),
        )
        self._conn.commit()
        return sid

    @_locked
    def list_sessions(self) -> list[SessionMeta]:
        rows = self._conn.execute(
            "SELECT s.id, s.title, s.created_at, COUNT(m.id) "
            "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
            "GROUP BY s.id ORDER BY s.created_at DESC, s.rowid DESC"
        ).fetchall()
        return [SessionMeta(r[0], r[1], r[2], r[3]) for r in rows]

    @_locked
    def delete_session(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._conn.execute(
            "DELETE FROM session_states WHERE session_id = ?", (session_id,)
        )  # M6 状态表同步清理，避免孤儿残留
        self._conn.execute(
            "DELETE FROM session_compacts WHERE session_id = ?", (session_id,)
        )  # M12 压缩记录同步清理
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    @_locked
    def get_title(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT title FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None

    @_locked
    def update_title(self, session_id: str, title: str) -> None:
        """M4：首条用户消息自动生成会话标题。"""
        self._conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
        )
        self._conn.commit()

    # -- 消息 ---------------------------------------------------------------

    @_locked
    def add_message(self, session_id: str, msg: Message) -> None:
        self._conn.execute(
            "INSERT INTO messages "
            "(session_id, role, content, reasoning_content, tool_calls, tool_call_id, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                msg.role,
                msg.content,
                msg.reasoning_content,
                _tool_calls_to_json(msg.tool_calls),
                msg.tool_call_id,
                _now(),
            ),
        )
        self._conn.commit()

    @_locked
    def add_messages(self, session_id: str, msgs: list[Message]) -> None:
        for m in msgs:
            self.add_message(session_id, m)

    @_locked
    def get_history(self, session_id: str) -> list[Message]:
        rows = self._conn.execute(
            "SELECT role, content, reasoning_content, tool_calls, tool_call_id FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [
            Message(r[0], r[1], r[2], _tool_calls_from_json(r[3]), r[4]) for r in rows
        ]

    # -- M6：任务计划消息与会话状态 -------------------------------------------

    @_locked
    def upsert_plan_message(self, session_id: str, text: str) -> None:
        """替换会话里的计划消息：历史中只保留最新一份（LIKE 匹配标记）。"""
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND role = 'system' "
            "AND content LIKE '%[vgent-plan]%'",
            (session_id,),
        )
        self.add_message(session_id, Message("system", text))

    @_locked
    def clear_plan(self, session_id: str) -> None:
        """/plan new：清掉计划消息，下次对话重新规划。"""
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND role = 'system' "
            "AND content LIKE '%[vgent-plan]%'",
            (session_id,),
        )
        self._conn.commit()

    @_locked
    def set_state(self, session_id: str, state: str) -> None:
        """落当前 Agent 状态（M6：每轮结束写一次，供恢复/展示）。"""
        self._conn.execute(
            "INSERT INTO session_states (session_id, state, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET state = excluded.state, "
            "updated_at = excluded.updated_at",
            (session_id, state, _now()),
        )
        self._conn.commit()

    @_locked
    def get_state(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT state FROM session_states WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None

    # -- M12：压缩记录（摘要 + 保留尾部 + 边界）--------------------------------

    @_locked
    def last_message_id(self, session_id: str) -> int | None:
        """该会话当前最后一条消息 id（压缩时刻的边界用）。"""
        row = self._conn.execute(
            "SELECT MAX(id) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    @_locked
    def upsert_compact(
        self, session_id: str, summary: str, retained: list[Message], boundary_id: int
    ) -> None:
        """写/覆盖该会话的压缩记录（只保留最新一份）。

        retained：压缩时保留的尾部消息（与 messages 表同源，序列化存 JSON）；
        boundary_id：压缩时刻的最后消息 id——之后新增的消息用 get_history_after 续接。
        """
        self._conn.execute(
            "INSERT INTO session_compacts "
            "(session_id, summary, retained, boundary_id, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, "
            "retained=excluded.retained, boundary_id=excluded.boundary_id, "
            "updated_at=excluded.updated_at",
            (
                session_id,
                summary,
                _messages_to_json(retained),
                boundary_id,
                _now(),
            ),
        )
        self._conn.commit()

    @_locked
    def get_compact(self, session_id: str) -> tuple[str, list[Message], int] | None:
        """读压缩记录 (摘要, 保留尾部, 边界 id)；无则 None。"""
        row = self._conn.execute(
            "SELECT summary, retained, boundary_id FROM session_compacts "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return row[0], _messages_from_json(row[1]), int(row[2])

    @_locked
    def get_history_after(self, session_id: str, message_id: int) -> list[Message]:
        """id > message_id 的消息（M12：压缩边界之后的新增消息，按序）。"""
        rows = self._conn.execute(
            "SELECT role, content, reasoning_content, tool_calls, tool_call_id FROM messages "
            "WHERE session_id = ? AND id > ? ORDER BY id",
            (session_id, message_id),
        ).fetchall()
        return [
            Message(r[0], r[1], r[2], _tool_calls_from_json(r[3]), r[4]) for r in rows
        ]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _tool_calls_to_json(calls: list[ToolCall] | None) -> str | None:
    if not calls:
        return None
    return json.dumps(
        [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in calls],
        ensure_ascii=False,
    )


def _tool_calls_from_json(raw: str | None) -> list[ToolCall] | None:
    if not raw:
        return None
    data = json.loads(raw)
    return [ToolCall(d["id"], d["name"], d.get("arguments", "")) for d in data]


def _messages_to_json(msgs: list[Message]) -> str:
    """压缩保留的尾部消息 → JSON（供 session_compacts.retained）。"""
    return json.dumps(
        [
            {
                "role": m.role,
                "content": m.content,
                "reasoning_content": m.reasoning_content,
                "tool_calls": (
                    [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in m.tool_calls]
                    if m.tool_calls
                    else None
                ),
                "tool_call_id": m.tool_call_id,
            }
            for m in msgs
        ],
        ensure_ascii=False,
    )


def _messages_from_json(raw: str | None) -> list[Message]:
    if not raw:
        return []
    data = json.loads(raw)
    out: list[Message] = []
    for d in data:
        tc = (
            [ToolCall(x["id"], x["name"], x.get("arguments", "")) for x in d["tool_calls"]]
            if d.get("tool_calls")
            else None
        )
        out.append(Message(d["role"], d["content"], d.get("reasoning_content"), tc, d.get("tool_call_id")))
    return out
