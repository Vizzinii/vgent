"""M1 测试：会话存储 CRUD（SQLite）。"""
from vgent.messages import Message, ToolCall
from vgent.store import SessionStore


def test_create_and_history_roundtrip(tmp_path) -> None:
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    store.add_message(sid, Message("user", "hi"))
    store.add_message(sid, Message("assistant", "hello"))
    hist = store.get_history(sid)
    assert [(m.role, m.content) for m in hist] == [("user", "hi"), ("assistant", "hello")]
    store.close()


def test_list_and_delete(tmp_path) -> None:
    store = SessionStore(tmp_path / "t.db")
    sid1 = store.create_session()
    store.create_session()  # 第二个会话只用于计数
    store.add_message(sid1, Message("user", "a"))
    sessions = store.list_sessions()
    assert len(sessions) == 2
    s1 = next(s for s in sessions if s.id == sid1)
    assert s1.message_count == 1
    store.delete_session(sid1)
    assert len(store.list_sessions()) == 1
    store.close()


def test_tool_calls_roundtrip(tmp_path) -> None:
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    tc = ToolCall(id="call_1", name="shell", arguments='{"cmd": "ls"}')
    store.add_message(sid, Message("assistant", "", tool_calls=[tc]))
    hist = store.get_history(sid)
    assert hist[0].tool_calls is not None
    assert hist[0].tool_calls[0].name == "shell"
    assert hist[0].tool_calls[0].arguments == '{"cmd": "ls"}'
    store.close()


def test_reasoning_content_roundtrip(tmp_path) -> None:
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    store.add_message(sid, Message("assistant", "", reasoning_content="思考中"))
    hist = store.get_history(sid)
    assert hist[0].reasoning_content == "思考中"
    store.close()


def test_plan_message_upsert_and_clear(tmp_path) -> None:
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    store.upsert_plan_message(sid, '[vgent-plan]\n{"steps": [{"description": "a"}]}\n[/vgent-plan]')
    store.upsert_plan_message(sid, '[vgent-plan]\n{"steps": [{"description": "b"}]}\n[/vgent-plan]')
    plans = [m for m in store.get_history(sid) if "[vgent-plan]" in m.content]
    assert len(plans) == 1  # 历史里只留最新一份
    assert "b" in plans[0].content
    store.clear_plan(sid)
    assert all("[vgent-plan]" not in m.content for m in store.get_history(sid))
    store.close()


def test_session_state_roundtrip(tmp_path) -> None:
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    assert store.get_state(sid) is None
    store.set_state(sid, "planning")
    assert store.get_state(sid) == "planning"
    store.set_state(sid, "completed")  # 覆盖
    assert store.get_state(sid) == "completed"
    store.close()


def test_update_title(tmp_path) -> None:
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    assert store.get_title(sid) == "新会话"
    store.update_title(sid, "帮我看看项目结构")
    assert store.get_title(sid) == "帮我看看项目结构"
    store.close()
