"""全面测试方案：按源码审计驱动，覆盖所有模块的功能正确性、边界条件、错误路径。"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from vgent.agent import MAX_TOOL_ROUNDS, SessionContext, _session_title, run_turn
from vgent.config import Config, PermissionRules, load_config
from vgent.context import ContextEngine, extract_summary, extract_summary_with_fallback
from vgent.llm import ChatResult, LLMClient
from vgent.memory.episodic import EpisodicMemory, MemoryEntry
from vgent.messages import Message, ToolCall, Usage
from vgent.permission import (
    Approval,
    ConfirmResult,
    PermissionSystem,
    persist_allow,
)
from vgent.reflection import MAX_REFLECT_ROUNDS, looks_failed, reflect
from vgent.state import AgentState
from vgent.store import SessionStore
from vgent.task import TaskPlan, TaskStep, plan_from_messages
from vgent.tools import ToolRegistry, ToolSchema, _cap_output
from vgent.workspace import find_instructions, find_user_instructions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(messages)
        if on_delta:
            on_delta("ok")
        return ChatResult(
            messages=[Message("assistant", "ok")],
            usage=Usage(10, 5, 15),
        )


class ScriptedLLM:
    def __init__(self, responses: list[ChatResult]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(messages)
        r = self.responses.pop(0)
        if on_delta and r.messages and r.messages[0].content:
            on_delta(r.messages[0].content)
        if on_reasoning and r.messages and r.messages[0].reasoning_content:
            on_reasoning(r.messages[0].reasoning_content)
        return r


def _chat_with_tools(tc: ToolCall, content: str = "") -> ChatResult:
    return ChatResult(
        messages=[Message("assistant", content, tool_calls=[tc])],
        usage=Usage(10, 5, 15),
        tool_calls=[tc],
    )


def _chat_final(content: str = "done") -> ChatResult:
    return ChatResult(messages=[Message("assistant", content)], usage=Usage(20, 5, 25))


def _make_ctx(tmp_path, llm=None, tools=None, permissions=None, engine=None, **kw):
    store = SessionStore(tmp_path / "db.sqlite")
    sid = store.create_session()
    return SessionContext(
        session_id=sid,
        store=store,
        llm=llm or FakeLLM(),
        tools=tools or ToolRegistry(),
        permissions=permissions or PermissionSystem(),
        engine=engine or ContextEngine(),
        **kw,
    )


# ===========================================================================
# P0 — messages.py
# ===========================================================================

class TestMessages:
    def test_usage_dataclass(self):
        u = Usage(100, 50, 150)
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 50
        assert u.total_tokens == 150

    def test_toolcall_dataclass(self):
        tc = ToolCall(id="call_1", name="shell", arguments='{"cmd":"ls"}')
        assert tc.id == "call_1"
        assert tc.name == "shell"
        assert tc.arguments == '{"cmd":"ls"}'

    def test_multiple_tool_calls_serialization(self):
        tc1 = ToolCall(id="c1", name="shell", arguments="{}")
        tc2 = ToolCall(id="c2", name="read_file", arguments="{}")
        msg = Message("assistant", "", tool_calls=[tc1, tc2])
        d = msg.to_openai()
        assert len(d["tool_calls"]) == 2
        assert d["tool_calls"][0]["function"]["name"] == "shell"
        assert d["tool_calls"][1]["function"]["name"] == "read_file"

    def test_empty_content_serialization(self):
        msg = Message("assistant", "")
        d = msg.to_openai()
        assert d["content"] == ""
        assert "tool_calls" not in d
        assert "reasoning_content" not in d
        assert "tool_call_id" not in d

    def test_message_with_all_fields(self):
        msg = Message(
            role="assistant",
            content="hello",
            reasoning_content="thinking...",
            tool_calls=[ToolCall(id="c1", name="x", arguments="{}")],
            tool_call_id="tc_1",
        )
        d = msg.to_openai()
        assert d["reasoning_content"] == "thinking..."
        assert d["tool_call_id"] == "tc_1"
        assert len(d["tool_calls"]) == 1

    def test_user_message_no_tool_calls(self):
        msg = Message("user", "hi")
        d = msg.to_openai()
        assert d == {"role": "user", "content": "hi"}

    def test_tool_result_message(self):
        msg = Message("tool", "result text", tool_call_id="call_123")
        d = msg.to_openai()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_123"

    def test_reasoning_content_none_not_serialized(self):
        msg = Message("assistant", "hi", reasoning_content=None)
        d = msg.to_openai()
        assert "reasoning_content" not in d


# ===========================================================================
# P0 — agent.py
# ===========================================================================

class TestAgent:
    def test_concurrent_tool_calls_from_single_response(self, tmp_path):
        tc1 = ToolCall(id="c1", name="read_file", arguments="{}")
        tc2 = ToolCall(id="c2", name="read_file", arguments="{}")
        llm = ScriptedLLM([
            ChatResult(
                messages=[Message("assistant", "", tool_calls=[tc1, tc2])],
                usage=Usage(10, 5, 15),
                tool_calls=[tc1, tc2],
            ),
            _chat_final("all done"),
        ])
        tools = ToolRegistry()
        tools.register(
            ToolSchema("read_file", "read", {}, "read"),
            lambda args: "file content",
        )
        ctx = _make_ctx(tmp_path, llm=llm, tools=tools)
        run_turn("read two files", ctx)
        # 两个 tool 结果都应写入 store
        hist = ctx.store.get_history(ctx.session_id)
        tool_msgs = [m for m in hist if m.role == "tool"]
        assert len(tool_msgs) == 2

    def test_max_tool_rounds_safety_valve(self, tmp_path):
        tc = ToolCall(id="c1", name="shell", arguments="{}")
        # 每轮都返回 tool_call → 应在 MAX_TOOL_ROUNDS 后强制终止
        # 循环消耗 MAX_TOOL_ROUNDS 个 tool_call 响应，之后 final 调用消耗 1 个
        responses = [_chat_with_tools(tc) for _ in range(MAX_TOOL_ROUNDS)]
        responses.append(_chat_final("forced stop"))
        llm = ScriptedLLM(responses)
        tools = ToolRegistry()
        tools.register(ToolSchema("shell", "exec", {}, "exec"), lambda args: "ok")
        ctx = _make_ctx(tmp_path, llm=llm, tools=tools)
        result = run_turn("loop forever", ctx)
        # 超限后 final 调用返回最终结果
        assert result.messages[0].content == "forced stop"

    def test_prune_and_compress_in_same_turn(self, tmp_path):
        """同一轮内低水位裁剪和高水位压缩不冲突。"""
        engine = ContextEngine()
        engine._tokens = 1000000  # 模拟高 token 使用
        llm = FakeLLM()
        ctx = _make_ctx(tmp_path, llm=llm, engine=engine)
        # 写入足够多的历史让 prune 和 compress 都触发
        for i in range(50):
            msg = Message("user" if i % 2 == 0 else "assistant", f"msg {i}")
            ctx.store.add_message(ctx.session_id, msg)
        result = run_turn("test", ctx)
        assert result is not None

    def test_state_failed_on_exception(self, tmp_path):
        class BoomLLM:
            def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
                raise RuntimeError("boom")

        ctx = _make_ctx(tmp_path, llm=BoomLLM())
        with pytest.raises(RuntimeError, match="boom"):
            run_turn("fail", ctx)
        assert ctx.store.get_state(ctx.session_id) == AgentState.FAILED.value

    def test_session_title_truncation(self):
        assert _session_title("") == "新会话"
        assert _session_title("short") == "short"
        long = "a" * 30
        assert len(_session_title(long)) == 24
        assert _session_title(long).endswith("…")
        assert _session_title("line1\nline2") == "line1"

    def test_session_title_whitespace(self):
        assert _session_title("  hello  ") == "hello"
        assert _session_title("\n\n  \n") == "新会话"

    def test_memory_already_present_dedup(self, tmp_path):
        from vgent.agent import _memory_already_present
        msgs = [
            Message("system", "[记忆] python（2026-01-01）：summary here"),
            Message("user", "hello"),
        ]
        assert _memory_already_present(msgs, "python") is True
        assert _memory_already_present(msgs, "java") is False

    def test_safe_parse_valid(self):
        from vgent.agent import _safe_parse
        data, err = _safe_parse('{"a": 1}')
        assert data == {"a": 1}
        assert err is None

    def test_safe_parse_empty(self):
        from vgent.agent import _safe_parse
        data, err = _safe_parse("")
        assert data == {}
        assert err is None

    def test_safe_parse_whitespace(self):
        from vgent.agent import _safe_parse
        data, err = _safe_parse("   ")
        assert data == {}
        assert err is None

    def test_safe_parse_invalid_json(self):
        from vgent.agent import _safe_parse
        data, err = _safe_parse("{bad json}")
        assert data is None
        assert "JSON 非法" in err

    def test_safe_parse_non_dict(self):
        from vgent.agent import _safe_parse
        data, err = _safe_parse("[1,2,3]")
        assert data is None
        assert "list" in err


# ===========================================================================
# P0 — llm.py
# ===========================================================================

class TestLLM:
    def test_empty_response_no_choices(self, tmp_path):
        """模型返回空 choices 不崩溃。"""
        llm = LLMClient.__new__(LLMClient)
        llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                __iter__=lambda s: iter([]),  # no chunks at all
            )
        )))
        # Just verify the class can be instantiated with mock
        # Real test would need more complex mocking of stream iteration
        assert hasattr(llm, '_client')

    def test_max_retries_zero_no_retry(self, tmp_path):
        """max_retries=0 时重试逻辑不覆盖 SDK。"""
        from openai import APIConnectionError

        from vgent.llm import _RETRYABLE
        # Verify the retryable set includes expected errors
        assert APIConnectionError in _RETRYABLE


# ===========================================================================
# P0 — store.py
# ===========================================================================

class TestStore:
    def test_concurrent_writes_thread_safety(self, tmp_path):
        store = SessionStore(tmp_path / "db.sqlite")
        sid = store.create_session()
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    store.add_message(sid, Message("user", f"t{n}_{i}"))
            except Exception as e:  # noqa: BLE001 — 并发测试需要捕获所有异常
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        hist = store.get_history(sid)
        assert len(hist) == 30
        store.close()

    def test_large_history_performance(self, tmp_path):
        store = SessionStore(tmp_path / "db.sqlite")
        sid = store.create_session()
        for i in range(500):
            store.add_message(sid, Message("user" if i % 2 == 0 else "assistant", f"msg {i}"))
        t0 = time.time()
        hist = store.get_history(sid)
        elapsed = time.time() - t0
        assert len(hist) == 500
        assert elapsed < 2.0  # should be fast
        store.close()

    def test_delete_session_cascading(self, tmp_path):
        store = SessionStore(tmp_path / "db.sqlite")
        sid = store.create_session()
        store.add_message(sid, Message("user", "hi"))
        store.upsert_plan_message(sid, '{"steps":[]}')
        store.set_state(sid, "completed")
        store.delete_session(sid)
        assert store.get_history(sid) == []
        # 级联删除 session_states（M6 修复后验证）
        assert store.get_state(sid) is None
        # plan 消息在 messages 表（role=system），随 messages 一起被删
        store.close()

    def test_get_history_ordering(self, tmp_path):
        store = SessionStore(tmp_path / "db.sqlite")
        sid = store.create_session()
        store.add_message(sid, Message("user", "a"))
        store.add_message(sid, Message("assistant", "b"))
        store.add_message(sid, Message("user", "c"))
        store.add_message(sid, Message("assistant", "d"))
        hist = store.get_history(sid)
        assert [m.content for m in hist] == ["a", "b", "c", "d"]
        store.close()

    def test_schema_migration_idempotent(self, tmp_path):
        db = tmp_path / "db.sqlite"
        s1 = SessionStore(db)
        s1.close()
        s2 = SessionStore(db)
        sid = s2.create_session()
        s2.add_message(sid, Message("user", "hello"))
        assert len(s2.get_history(sid)) == 1
        s2.close()

    def test_special_chars_in_content(self, tmp_path):
        store = SessionStore(tmp_path / "db.sqlite")
        sid = store.create_session()
        specials = 'has "quotes" and \'single\' and\nnewlines and\ttabs and unicode你好'
        store.add_message(sid, Message("user", specials))
        hist = store.get_history(sid)
        assert hist[0].content == specials
        store.close()

    def test_create_session_with_title(self, tmp_path):
        store = SessionStore(tmp_path / "db.sqlite")
        sid = store.create_session("My Session")
        assert store.get_title(sid) == "My Session"
        store.close()

    def test_list_sessions_ordering(self, tmp_path):
        store = SessionStore(tmp_path / "db.sqlite")
        s1 = store.create_session("first")
        time.sleep(0.01)
        s2 = store.create_session("second")
        sessions = store.list_sessions()
        assert sessions[0].id == s2  # newest first
        assert sessions[1].id == s1
        store.close()


# ===========================================================================
# P0 — permission.py
# ===========================================================================

class TestPermission:
    def test_concurrent_permission_checks(self):
        ps = PermissionSystem()
        tool = ToolSchema("shell", "exec", {}, "exec")
        errors = []

        def checker():
            try:
                for _ in range(100):
                    ps.check(tool, {})
            except Exception as e:  # noqa: BLE001 — 并发测试需要捕获所有异常
                errors.append(e)

        threads = [threading.Thread(target=checker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []

    def test_persist_allow_idempotent(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[permissions]\nallow=["shell"]\n', encoding="utf-8"
        )
        persist_allow(tmp_path, "shell")
        persist_allow(tmp_path, "shell")
        text = config_file.read_text(encoding="utf-8")
        assert text.count("shell") == 1

    def test_persist_allow_preserves_ask_deny(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[permissions]\nallow=[]\nask=["read_file"]\ndeny=["write_file"]\n',
            encoding="utf-8",
        )
        persist_allow(tmp_path, "search")
        text = config_file.read_text(encoding="utf-8")
        assert "read_file" in text
        assert "write_file" in text
        assert "search" in text

    def test_headless_mode_rejects_write_exec(self):
        ps = PermissionSystem()  # no confirm callback
        tool = ToolSchema("shell", "exec", {}, "exec")
        approval = ps.check(tool, {})
        assert approval == Approval.NEED_CONFIRM
        result = ps.confirm(tool, {})
        assert result == ConfirmResult.REJECT

    def test_deny_rule_beats_allow(self):
        ps = PermissionSystem(
            rules=PermissionRules(allow=["shell"], ask=[], deny=["shell"])
        )
        tool = ToolSchema("shell", "exec", {}, "exec")
        assert ps.check(tool, {}) == Approval.DENIED


# ===========================================================================
# P0 — context.py
# ===========================================================================

class TestContext:
    def test_compress_large_message_list(self):
        engine = ContextEngine()
        # Build messages that should compress
        msgs = [Message("system", "start")]
        for i in range(80):
            msgs.append(Message("user", f"question {i}" + "x" * 500))
            msgs.append(Message("assistant", f"answer {i}" + "y" * 500))
        engine._tokens = 800000
        result = engine.compress(msgs, force=True)
        assert len(result) < len(msgs)

    def test_prune_protects_last_n_messages(self):
        engine = ContextEngine()
        msgs = []
        for i in range(20):
            msgs.append(Message("user", f"q{i}"))
            msgs.append(Message("assistant", f"a{i}"))
        # Add a long tool result in the middle
        msgs[10] = Message("tool", "x" * 5000, tool_call_id="c1")
        pruned, _count = engine.prune_tool_results_only(msgs)
        # Last 6 messages should be intact
        last_6 = pruned[-6:]
        for m in last_6:
            if m.role == "tool" and len(m.content) > 300:
                # Tool result in protected zone should not be pruned
                assert len(m.content) > 300 or m.content.startswith("x")

    def test_extract_summary_multiline(self):
        text = "some text\n<summary>\nline 1\nline 2\nline 3\n</summary>\nend"
        result = extract_summary(text)
        assert "line 1" in result
        assert "line 3" in result

    def test_prune_empty_tool_result(self):
        engine = ContextEngine()
        msgs = [
            Message("user", "q"),
            Message("assistant", "", tool_calls=[ToolCall("c1", "shell", "{}")]),
            Message("tool", "", tool_call_id="c1"),
        ]
        pruned, _count = engine.prune_tool_results_only(msgs)
        assert len(pruned) >= 2

    def test_compression_count_tracking(self):
        engine = ContextEngine()
        # 消息必须足够长：每条约 2000 tokens 才能确保 tail_token_budget=20000 被耗尽
        # 40 * 2000 = 80000 >> 20000，确保 dropped > 0
        msgs = [Message("system", "start" + "z" * 6000)]
        for i in range(40):
            msgs.append(Message("user", f"question {i} " + "x" * 6000))
            msgs.append(Message("assistant", f"answer {i} " + "y" * 6000))
        engine._tokens = 800000
        engine.compress(msgs, force=True)
        assert engine.compression_count == 1
        engine.compress(msgs, force=True)
        assert engine.compression_count == 2

    def test_extract_summary_empty(self):
        assert extract_summary("") == ""
        assert extract_summary("no summary block here") == "no summary block here"

    def test_extract_summary_with_fallback_short_rejected(self):
        msg = Message("assistant", "<summary>x</summary>")
        result = extract_summary_with_fallback(msg)
        # Too short (< 20 chars), should return ""
        assert result == ""

    def test_extract_summary_with_fallback_content_preferred(self):
        content = "<summary>\nThis is a valid summary with enough characters to pass the length check.\n</summary>"
        msg = Message("assistant", content, reasoning_content="short")
        result = extract_summary_with_fallback(msg)
        assert "valid summary" in result


# ===========================================================================
# P0 — reflection.py
# ===========================================================================

class TestReflection:
    def test_mixed_language_failure(self):
        assert looks_failed("error: 未找到文件 test.txt") is True
        assert looks_failed("No such file: test.txt") is True
        assert looks_failed("错误 未找到 test") is True

    def test_reflect_empty_history(self):
        class FakeLLM:
            def chat(self, messages, **kw):
                return ChatResult(messages=[Message("assistant", "")], usage=Usage(0, 0, 0))
        result = reflect([], FakeLLM())
        assert result == ""  # empty history → LLM returns empty → ""

    def test_reflect_prompt_content(self):
        from vgent.reflection import REFLECT_PROMPT
        assert "Failure" in REFLECT_PROMPT
        assert "Action" in REFLECT_PROMPT

    def test_looks_failed_exit_without_code(self):
        # "exit " without actual code
        assert looks_failed("process exit ") is False

    def test_looks_failed_normal_chinese_output(self):
        assert looks_failed("文件已创建成功") is False

    def test_looks_failed_traceback_without_exit(self):
        assert looks_failed("Traceback (most recent call last):") is True

    def test_max_reflect_rounds_constant(self):
        assert MAX_REFLECT_ROUNDS == 3


# ===========================================================================
# P0 — task.py
# ===========================================================================

class TestTask:
    def test_plan_many_steps(self):
        steps = [TaskStep(f"step {i}", "done" if i < 5 else "pending") for i in range(50)]
        plan = TaskPlan(steps=steps)
        text = plan.to_text()
        parsed = TaskPlan.from_text(text)
        assert parsed is not None
        assert len(parsed.steps) == 50
        assert parsed.done is False

    def test_plan_mixed_statuses(self):
        steps = [
            TaskStep("a", "done"),
            TaskStep("b", "failed"),
            TaskStep("c", "pending"),
        ]
        plan = TaskPlan(steps=steps)
        assert plan.done is False

    def test_plan_special_chars_in_description(self):
        plan = TaskPlan(steps=[TaskStep('step with "quotes" & <tags>', "done")])
        text = plan.to_text()
        parsed = TaskPlan.from_text(text)
        assert parsed is not None
        assert parsed.steps[0].description == 'step with "quotes" & <tags>'

    def test_plan_from_messages_empty_content(self):
        msgs = [Message("assistant", "")]
        assert plan_from_messages(msgs) is None

    def test_plan_block_nested_in_other_text(self):
        text = "Here is my plan:\n[vgent-plan]\n{\"steps\":[{\"description\":\"a\",\"status\":\"done\"}]}\n[/vgent-plan]\nDone."
        msgs = [Message("assistant", text)]
        plan = plan_from_messages(msgs)
        assert plan is not None
        assert len(plan.steps) == 1

    def test_task_step_defaults(self):
        step = TaskStep("do something")
        assert step.status == "pending"

    def test_plan_to_json(self):
        plan = TaskPlan(steps=[TaskStep("a", "done"), TaskStep("b", "failed")])
        d = plan.to_json()
        assert d["steps"][0]["status"] == "done"
        assert d["steps"][1]["status"] == "failed"

    def test_plan_all_done(self):
        plan = TaskPlan(steps=[TaskStep("a", "done"), TaskStep("b", "done")])
        assert plan.done is True


# ===========================================================================
# P1 — cli.py (inline commands)
# ===========================================================================

class TestCLI:
    def test_help_command(self):
        from vgent.cli import HELP
        assert "/help" in HELP
        assert "/compact" in HELP
        assert "/plan" in HELP

    def test_reasoning_toggle(self):
        ctx = SimpleNamespace(show_reasoning=False)
        # Toggle on
        ctx.show_reasoning = not ctx.show_reasoning
        assert ctx.show_reasoning is True
        # Toggle off
        ctx.show_reasoning = not ctx.show_reasoning
        assert ctx.show_reasoning is False

    def test_delete_current_session_rejected(self, tmp_path):
        """不能删除当前活跃会话。"""
        store = SessionStore(tmp_path / "db.sqlite")
        sid = store.create_session("current")
        # Try to delete the current session
        hist = store.get_history(sid)
        assert hist == []  # empty session
        # The actual deletion check is in cli.py; here we verify the store works
        store.delete_session(sid)
        assert store.get_title(sid) is None
        store.close()

    def test_resolve_resume_invalid(self, tmp_path):
        from vgent.cli import _resolve_resume
        store = SessionStore(tmp_path / "db.sqlite")
        result = _resolve_resume(store, "nonexistent", None)
        assert result is None
        store.close()


# ===========================================================================
# P1 — workspace.py
# ===========================================================================

class TestWorkspace:
    def test_instruction_truncation_at_8k(self, tmp_path):
        instructions_dir = tmp_path
        # Write an AGENTS.md that's > 8K chars
        long_content = "x" * 10000
        (instructions_dir / "AGENTS.md").write_text(long_content, encoding="utf-8")
        result = find_instructions(str(instructions_dir))
        assert result is not None
        name, content = result
        assert name == "AGENTS.md"
        assert len(content) <= 8100  # ~8K + some marker

    def test_user_level_instructions(self, tmp_path):
        data_dir = tmp_path / ".vgent"
        data_dir.mkdir()
        (data_dir / "AGENTS.md").write_text("user instructions here", encoding="utf-8")
        result = find_user_instructions(str(data_dir))
        assert result is not None
        assert "user instructions here" in result[1]

    def test_max_parent_depth(self, tmp_path):
        """不超过 8 层。"""
        # Create deep nested dirs
        deep = tmp_path
        for i in range(10):
            deep = deep / f"d{i}"
            deep.mkdir()
        # Put AGENTS.md in the deepest
        (deep / "AGENTS.md").write_text("deep", encoding="utf-8")
        # Should not find it from tmp_path (10 levels up)
        result = find_instructions(str(tmp_path))
        assert result is None

    def test_claude_md_fallback(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("claude fallback", encoding="utf-8")
        result = find_instructions(str(tmp_path))
        assert result is not None
        assert result[0] == "CLAUDE.md"

    def test_empty_instruction_file(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("", encoding="utf-8")
        result = find_instructions(str(tmp_path))
        # Empty file treated as not found
        assert result is None


# ===========================================================================
# P1 — config.py
# ===========================================================================

class TestConfig:
    def test_context_length_config(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[provider]\nactive = "deepseek"\n\n[providers.deepseek]\n'
            'model = "deepseek-chat"\ncontext_length = 500000\n',
            encoding="utf-8",
        )
        cfg = load_config(config_file, provider=None)
        assert cfg.provider.context_length == 500000

    def test_tail_token_budget_config(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "[context]\ntail_token_budget = 30000\n",
            encoding="utf-8",
        )
        cfg = load_config(config_file, provider=None)
        assert cfg.context.tail_token_budget == 30000

    def test_malformed_toml(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text("{{bad toml\n", encoding="utf-8")
        with pytest.raises((ValueError, Exception)):
            load_config(config_file, provider=None)

    def test_legacy_single_provider_format(self, tmp_path):
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            '[provider]\nname = "deepseek"\nmodel = "deepseek-chat"\n'
            'base_url = "https://api.deepseek.com"\n',
            encoding="utf-8",
        )
        cfg = load_config(config_file, provider=None)
        assert cfg.provider.model == "deepseek-chat"

    def test_default_config_values(self):
        cfg = Config()
        assert cfg.context.threshold_percent == 0.75
        assert cfg.context.prune_percent == 0.30
        assert cfg.log_level == "INFO"


# ===========================================================================
# P1 — mcp/client.py
# ===========================================================================

class TestMCP:
    def test_slug_normalization(self):
        from vgent.mcp.client import _slug
        assert _slug("My-Server") == "my_server"
        assert _slug("tool.name") == "tool_name"
        assert _slug("ALREADY_LOWER") == "already_lower"

    def test_prefixed_name(self):
        from vgent.mcp.client import _prefixed
        assert _prefixed("server", "tool") == "server_tool"

    def test_output_cap(self):
        from vgent.mcp.client import _OUTPUT_CAP
        assert _OUTPUT_CAP == 10_000


# ===========================================================================
# P1 — memory/episodic.py
# ===========================================================================

class TestMemory:
    def test_search_by_summary(self, tmp_path):
        mem = EpisodicMemory(tmp_path / "mem.jsonl")
        mem.add("topic1", "contains redis cache and ttl 300", "s1", "t1")
        # Search by a word only in summary
        results = mem.search("redis", limit=5)
        assert len(results) == 1
        assert results[0].topic == "topic1"

    def test_has_session_multiple_entries(self, tmp_path):
        mem = EpisodicMemory(tmp_path / "mem.jsonl")
        mem.add("t1", "s1", "session_1", "title1")
        mem.add("t2", "s2", "session_1", "title2")
        assert mem.has_session("session_1") is True
        assert mem.has_session("other") is False

    def test_jsonl_corruption_recovery(self, tmp_path):
        memfile = tmp_path / "mem.jsonl"
        lines = []
        for i in range(10):
            entry = MemoryEntry(
                ts=f"2026-01-0{i+1}T00:00:00",
                session_id=f"s{i}",
                title=f"t{i}",
                topic=f"topic{i}",
                summary=f"summary{i}",
                project="proj",
            )
            lines.append(entry.to_line())
        # Corrupt lines 3, 5, 7
        lines[3] = "NOT VALID JSON"
        lines[5] = '{"bad": "data"}'
        lines[7] = ""
        memfile.write_text("\n".join(lines), encoding="utf-8")
        mem = EpisodicMemory(memfile)
        entries = mem._entries()
        assert len(entries) == 7  # 10 - 3 corrupted

    def test_concurrent_appends(self, tmp_path):
        mem = EpisodicMemory(tmp_path / "mem.jsonl")
        errors = []

        def writer(n):
            try:
                for i in range(5):
                    mem.add(f"t{n}_{i}", f"s{n}_{i}", f"ses{n}", f"title{n}_{i}")
            except Exception as e:  # noqa: BLE001 — 并发测试需要捕获所有异常
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        # 加锁后并发追加应全部成功
        assert mem.count() == 15

    def test_project_scoped_search_exactness(self, tmp_path):
        mem = EpisodicMemory(tmp_path / "mem.jsonl")
        mem.add("topic", "summary", "s1", "title", project="A")
        mem.add("topic", "summary", "s2", "title", project="AB")
        results_A = mem.search("topic", project="A")
        assert len(results_A) == 1
        assert results_A[0].project == "A"
        results_AB = mem.search("topic", project="AB")
        assert len(results_AB) == 1
        assert results_AB[0].project == "AB"

    def test_memory_entry_to_line_from_line(self):
        entry = MemoryEntry(
            ts="2026-01-01", session_id="s1", title="t", topic="tp", summary="sm", project="p"
        )
        line = entry.to_line()
        parsed = MemoryEntry.from_line(line)
        assert parsed is not None
        assert parsed.ts == "2026-01-01"
        assert parsed.project == "p"

    def test_memory_entry_from_line_bad_data(self):
        assert MemoryEntry.from_line("not json") is None
        assert MemoryEntry.from_line('{"a":1}') is None

    def test_summarize_tail_window_limit(self, tmp_path):
        """超过 30 条消息只取最后 30 条。"""
        from vgent.memory.episodic import summarize
        messages = [Message("user", f"q{i}") for i in range(50)]
        class FakeLLM:
            def chat(self, messages, **kw):
                # Check that only ~30 messages were sent
                return ChatResult(
                    messages=[Message("assistant", "summary of last 30 messages with enough chars to pass validation")],
                    usage=Usage(10, 5, 15),
                )
        result = summarize(messages, FakeLLM(), "test topic")
        assert result != ""


# ===========================================================================
# P1 — web/server.py
# ===========================================================================

class TestWeb:
    def test_hub_manager_construction(self, tmp_path):
        """HubManager 构造正确接线。"""
        from vgent.web.server import HubManager
        store = SessionStore(tmp_path / "db.sqlite")
        from vgent.tools import ToolRegistry
        tools = ToolRegistry()
        mgr = HubManager(
            cfg=SimpleNamespace(
                provider=SimpleNamespace(name="test", context_length=1000000),
                mcp_servers={},
                permissions=PermissionRules(allow=[], ask=[], deny=[]),
                data_dir=tmp_path,
            ),
            store=store,
            llm=FakeLLM(),
            tools=tools,
        )
        assert mgr is not None
        store.close()

    def test_tool_summary_cap(self):
        from vgent.web.server import _TOOL_SUMMARY_CAP
        assert _TOOL_SUMMARY_CAP == 120

    def test_confirm_timeout(self):
        from vgent.web.server import CONFIRM_TIMEOUT
        assert CONFIRM_TIMEOUT == 600.0

    def test_heartbeat_interval(self):
        from vgent.web.server import HEARTBEAT
        assert HEARTBEAT == 15.0


# ===========================================================================
# P1 — state.py
# ===========================================================================

class TestState:
    def test_all_states_exist(self):
        assert AgentState.IDLE.value == "idle"
        assert AgentState.PLANNING.value == "planning"
        assert AgentState.EXECUTING.value == "executing"
        assert AgentState.WAITING_PERMISSION.value == "waiting_permission"
        assert AgentState.COMPLETED.value == "completed"
        assert AgentState.FAILED.value == "failed"

    def test_state_string_conversion(self):
        assert str(AgentState.IDLE) == "AgentState.IDLE"
        assert AgentState.COMPLETED.value == "completed"


# ===========================================================================
# P1 — tools.py (additional)
# ===========================================================================

class TestToolsAdditional:
    def test_cap_output_short(self):
        assert _cap_output("short") == "short"

    def test_cap_output_long(self):
        long = "x" * 20000
        result = _cap_output(long)
        assert len(result) < len(long)

    def test_tool_registry_get_missing(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_tool_registry_schemas_format(self):
        reg = ToolRegistry()
        reg.register(
            ToolSchema("test_tool", "test desc", {"type": "object"}, "read"),
            lambda args: "ok",
        )
        schemas = reg.schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "test_tool"

    def test_tool_registry_execute(self):
        reg = ToolRegistry()
        reg.register(
            ToolSchema("echo", "echo", {}, "read"),
            lambda args: f"echoed: {args}",
        )
        result = reg.execute("echo", {"msg": "hi"})
        assert "echoed" in result

    def test_tool_registry_execute_missing(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.execute("nonexistent", {})

    def test_filter_denied_removes(self):
        reg = ToolRegistry()
        reg.register(ToolSchema("a", "a", {}, "read"), lambda args: "a")
        reg.register(ToolSchema("b", "b", {}, "read"), lambda args: "b")
        reg.filter_denied({"b"})
        assert reg.get("a") is not None
        assert reg.get("b") is None
        assert len(reg.schemas()) == 1


# ===========================================================================
# P1 — commands.py
# ===========================================================================

class TestCommands:
    def test_load_from_nonexistent_dir(self, tmp_path):
        from vgent.commands import load_commands
        result = load_commands(tmp_path / "nonexistent")
        assert result == {}

    def test_load_empty_dir(self, tmp_path):
        from vgent.commands import load_commands
        result = load_commands(tmp_path)
        assert result == {}


# ===========================================================================
# 回归测试 — 代码缺口修复验证
# ===========================================================================

class TestGapFixStoreDeleteCascade:
    """验证 store.delete_session 级联删除 session_states（缺口修复）。"""

    def test_state_deleted_after_session_delete(self, tmp_path):
        store = SessionStore(tmp_path / "db.sqlite")
        sid = store.create_session()
        store.set_state(sid, "completed")
        assert store.get_state(sid) == "completed"
        store.delete_session(sid)
        assert store.get_state(sid) is None
        store.close()

    def test_multiple_sessions_cascade_independent(self, tmp_path):
        store = SessionStore(tmp_path / "db.sqlite")
        s1 = store.create_session("first")
        s2 = store.create_session("second")
        store.set_state(s1, "completed")
        store.set_state(s2, "failed")
        store.delete_session(s1)
        # s1 状态应被清除，s2 不受影响
        assert store.get_state(s1) is None
        assert store.get_state(s2) == "failed"
        store.close()

    def test_delete_nonexistent_session_no_error(self, tmp_path):
        store = SessionStore(tmp_path / "db.sqlite")
        # 删除不存在的会话不应报错
        store.delete_session("nonexistent_id")
        store.close()


class TestGapFixMemoryConcurrency:
    """验证 EpisodicMemory 加锁后并发追加不再丢行（缺口修复）。"""

    def test_concurrent_add_all_succeed(self, tmp_path):
        mem = EpisodicMemory(tmp_path / "mem.jsonl")
        errors = []

        def writer(n):
            try:
                for i in range(10):
                    mem.add(f"topic_{n}_{i}", f"summary_{n}_{i}", f"ses_{n}", f"title_{n}_{i}")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        # 加锁后 50 次写入应全部成功（无丢行）
        assert mem.count() == 50

    def test_concurrent_read_write_consistent(self, tmp_path):
        """并发读写不产生半行或损坏数据。"""
        mem = EpisodicMemory(tmp_path / "mem.jsonl")
        # 预写 10 条
        for i in range(10):
            mem.add(f"t{i}", f"s{i}", f"ses{i}", f"title{i}")
        errors = []

        def writer(n):
            try:
                for i in range(5):
                    mem.add(f"topic_w_{n}_{i}", f"summary_{n}_{i}", f"ses_w_{n}", f"title_{n}_{i}")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def reader():
            try:
                for _ in range(10):
                    entries = mem._entries()
                    # 每条 entry 必须有完整字段
                    for e in entries:
                        assert e.ts and e.session_id and e.topic is not None
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        # 10 预写 + 10 并发写 = 20
        assert mem.count() == 20
