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


def test_delete_session_cleans_states(tmp_path) -> None:
    """修复：删除会话同步清理 session_states，不留孤儿状态行。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    store.set_state(sid, "completed")
    assert store.get_state(sid) == "completed"
    store.delete_session(sid)
    assert store.get_state(sid) is None
    store.close()


# -- M12：压缩记录（摘要 + 保留尾部 + 边界） --------------------------------------


def test_compact_record_upsert_and_get(tmp_path) -> None:
    """M12：压缩记录 upsert 只留最新一份；保留尾部消息序列化往返。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    store.add_message(sid, Message("user", "a"))
    store.add_message(sid, Message("user", "b"))
    store.upsert_compact(sid, "摘要一", [Message("user", "b")], boundary_id=2)
    store.upsert_compact(sid, "摘要二", [], boundary_id=3)
    comp = store.get_compact(sid)
    assert comp is not None
    summary, retained, boundary = comp
    assert summary == "摘要二"
    assert retained == [] and boundary == 3
    store.close()


def test_compact_record_retained_roundtrip_with_tool_calls(tmp_path) -> None:
    """M12：保留尾部含 tool_calls / reasoning_content 时往返不丢。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    retained = [
        Message("assistant", "", reasoning_content="思考", tool_calls=[ToolCall("c1", "shell", '{"x":1}')]),
        Message("tool", "结果", tool_call_id="c1"),
    ]
    store.upsert_compact(sid, "摘要", retained, boundary_id=0)
    comp = store.get_compact(sid)
    assert comp is not None
    _summary, loaded, _boundary = comp
    assert loaded[0].reasoning_content == "思考"
    assert loaded[0].tool_calls[0].name == "shell"
    assert loaded[1].tool_call_id == "c1"
    store.close()


def test_get_history_after_boundary(tmp_path) -> None:
    """M12：boundary 之后的消息按序返回（不含 boundary 自身）。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    for c in "abc":
        store.add_message(sid, Message("user", c))
    store.upsert_compact(sid, "摘要", [], boundary_id=1)
    store.add_message(sid, Message("assistant", "d"))
    after = store.get_history_after(sid, 1)
    assert [(m.role, m.content) for m in after] == [("user", "b"), ("user", "c"), ("assistant", "d")]
    store.close()


def test_last_message_id(tmp_path) -> None:
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    assert store.last_message_id(sid) is None
    store.add_message(sid, Message("user", "x"))
    assert store.last_message_id(sid) == 1
    store.close()


def test_delete_session_cleans_compact(tmp_path) -> None:
    """M12：删除会话同步清理压缩记录。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    store.add_message(sid, Message("user", "x"))
    store.upsert_compact(sid, "摘要", [], boundary_id=1)
    assert store.get_compact(sid) is not None
    store.delete_session(sid)
    assert store.get_compact(sid) is None
    store.close()
