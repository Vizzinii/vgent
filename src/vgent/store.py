"""④ 会话存储：SQLite（双表 + thread_id，WAL）。

蓝本：openai-agents SQLiteSession（agent_sessions + agent_messages 双表、WAL）；
thread_id 主键概念取自 langgraph。路径在本机 ~/.vgent/sessions/，不进同步盘（决策 7）。
"""
from __future__ import annotations

import json
import sqlite3
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


class SessionStore:
    """vgent 会话库。

    两张表：
      sessions(id TEXT PRIMARY KEY, title, created_at)
      messages(id AUTOINCREMENT, session_id, role, content, tool_calls, tool_call_id, ts)
    全量历史留库（决策 8 ⑤：压缩只影响发送列表，真相永远在库里）。
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
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
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- 会话 CRUD ---------------------------------------------------------

    def create_session(self, title: str = "新会话") -> str:
        sid = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
            (sid, title, _now()),
        )
        self._conn.commit()
        return sid

    def list_sessions(self) -> list[SessionMeta]:
        rows = self._conn.execute(
            "SELECT s.id, s.title, s.created_at, COUNT(m.id) "
            "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
            "GROUP BY s.id ORDER BY s.created_at DESC, s.rowid DESC"
        ).fetchall()
        return [SessionMeta(r[0], r[1], r[2], r[3]) for r in rows]

    def delete_session(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    def get_title(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT title FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None

    def update_title(self, session_id: str, title: str) -> None:
        """M4：首条用户消息自动生成会话标题。"""
        self._conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
        )
        self._conn.commit()

    # -- 消息 ---------------------------------------------------------------

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

    def add_messages(self, session_id: str, msgs: list[Message]) -> None:
        for m in msgs:
            self.add_message(session_id, m)

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

    def upsert_plan_message(self, session_id: str, text: str) -> None:
        """替换会话里的计划消息：历史中只保留最新一份（LIKE 匹配标记）。"""
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND role = 'system' "
            "AND content LIKE '%[vgent-plan]%'",
            (session_id,),
        )
        self.add_message(session_id, Message("system", text))

    def clear_plan(self, session_id: str) -> None:
        """/plan new：清掉计划消息，下次对话重新规划。"""
        self._conn.execute(
            "DELETE FROM messages WHERE session_id = ? AND role = 'system' "
            "AND content LIKE '%[vgent-plan]%'",
            (session_id,),
        )
        self._conn.commit()

    def set_state(self, session_id: str, state: str) -> None:
        """落当前 Agent 状态（M6：每轮结束写一次，供恢复/展示）。"""
        self._conn.execute(
            "INSERT INTO session_states (session_id, state, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET state = excluded.state, "
            "updated_at = excluded.updated_at",
            (session_id, state, _now()),
        )
        self._conn.commit()

    def get_state(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT state FROM session_states WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None


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
