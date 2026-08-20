"""M1/M2 测试：agent 主循环（FakeLLM/ScriptedLLM 注入，不触网）。"""
from __future__ import annotations

import pytest

from vgent.agent import SessionContext, run_turn
from vgent.config import ContextConfig
from vgent.context import ContextEngine
from vgent.llm import ChatResult
from vgent.messages import Message, ToolCall, Usage
from vgent.permission import ConfirmResult, PermissionSystem
from vgent.state import AgentState
from vgent.store import SessionStore
from vgent.tools import ToolRegistry, ToolSchema


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(messages)
        if on_delta:
            on_delta("你好")
        return ChatResult(
            messages=[Message("assistant", "你好")],
            usage=Usage(10, 5, 15),
        )


class ScriptedLLM:
    """按顺序返回预设响应的假 LLM。"""

    def __init__(self, responses: list[ChatResult]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(messages)
        r = self.responses.pop(0)
        if on_delta and r.messages and r.messages[0].content:
            on_delta(r.messages[0].content)
        return r


def _chat_with_tools(tc: ToolCall, content: str = "") -> ChatResult:
    return ChatResult(
        messages=[Message("assistant", content, tool_calls=[tc])],
        usage=Usage(10, 5, 15),
        tool_calls=[tc],
    )


def _chat_final(content: str) -> ChatResult:
    return ChatResult(messages=[Message("assistant", content)], usage=Usage(20, 5, 25))


def test_run_turn_persists_and_returns(tmp_path) -> None:
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    llm = FakeLLM()
    ctx = SessionContext(session_id=sid, store=store, llm=llm)

    result = run_turn("帮我写段代码", ctx)
    assert result.messages[0].content == "你好"
    assert result.usage is not None and result.usage.total_tokens == 15

    hist = store.get_history(sid)
    assert [m.role for m in hist] == ["user", "assistant"]
    assert hist[0].content == "帮我写段代码"
    assert hist[1].content == "你好"

    # 第二轮应带上完整历史（M6：无计划时首条为规划提示 system 消息）
    run_turn("继续", ctx)
    assert [m.role for m in llm.calls[1]] == ["system", "user", "assistant", "user"]
    store.close()


def test_tool_loop_executes_read_tool(tmp_path) -> None:
    """read 档自动放行：工具执行、结果回喂、最终回答落库。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    reg = ToolRegistry()
    executed: list[dict] = []

    def add_handler(args: dict) -> str:
        executed.append(args)
        return str(args["a"] + args["b"])

    reg.register(ToolSchema("add", "加法", {"type": "object"}, "read"), add_handler)
    llm = ScriptedLLM(
        [
            _chat_with_tools(ToolCall("c1", "add", '{"a": 1, "b": 2}')),
            _chat_final("结果是 3"),
        ]
    )
    ctx = SessionContext(session_id=sid, store=store, llm=llm, tools=reg)

    result = run_turn("1+2=?", ctx)
    assert executed == [{"a": 1, "b": 2}]
    assert result.messages[0].content == "结果是 3"
    hist = store.get_history(sid)
    assert [m.role for m in hist] == ["user", "assistant", "tool", "assistant"]
    assert hist[2].content == "3"
    # 第二轮 LLM 调用应带上 tool 结果
    assert llm.calls[1][-1].role == "tool"
    store.close()


def test_tool_rejected_by_permission(tmp_path) -> None:
    """exec 档被拒：handler 不执行，拒绝消息回喂模型。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    reg = ToolRegistry()
    executed = False

    def rm_handler(args: dict) -> str:
        nonlocal executed
        executed = True
        return "deleted"

    reg.register(ToolSchema("rm", "删除", {"type": "object"}, "exec"), rm_handler)
    ps = PermissionSystem(confirm=lambda tool, args: ConfirmResult.REJECT)
    llm = ScriptedLLM(
        [
            _chat_with_tools(ToolCall("c1", "rm", '{"path": "/x"}')),
            _chat_final("已取消"),
        ]
    )
    ctx = SessionContext(session_id=sid, store=store, llm=llm, tools=reg, permissions=ps)

    run_turn("删掉 /x", ctx)
    assert executed is False
    tool_msg = next(m for m in store.get_history(sid) if m.role == "tool")
    assert "拒绝" in tool_msg.content
    store.close()


def test_unknown_tool_fed_back(tmp_path) -> None:
    """未知工具：不崩溃，错误回喂模型。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    llm = ScriptedLLM(
        [
            _chat_with_tools(ToolCall("c1", "no_such_tool", "{}")),
            _chat_final("完成"),
        ]
    )
    ctx = SessionContext(session_id=sid, store=store, llm=llm)

    run_turn("x", ctx)
    tool_msg = next(m for m in store.get_history(sid) if m.role == "tool")
    assert "未知工具" in tool_msg.content
    store.close()


def test_malformed_args_not_executed(tmp_path) -> None:
    """坏参数（决策 9）：不执行，解析错误回喂模型。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    reg = ToolRegistry()
    executed = False

    def add_handler(args: dict) -> str:
        nonlocal executed
        executed = True
        return "ok"

    reg.register(ToolSchema("add", "加法", {"type": "object"}, "read"), add_handler)
    llm = ScriptedLLM(
        [
            _chat_with_tools(ToolCall("c1", "add", "{bad json")),
            _chat_final("修正后完成"),
        ]
    )
    ctx = SessionContext(session_id=sid, store=store, llm=llm, tools=reg)

    run_turn("算一下", ctx)
    assert executed is False
    tool_msg = next(m for m in store.get_history(sid) if m.role == "tool")
    assert "参数解析失败" in tool_msg.content
    store.close()


def test_sticky_confirm_auto_second_call(tmp_path) -> None:
    """ALWAYS 确认后本会话 sticky：第二次调用不再问。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    reg = ToolRegistry()
    confirm_count: list[int] = []
    executed = 0

    def run_handler(args: dict) -> str:
        nonlocal executed
        executed += 1
        return "ok"

    def cb(tool, args):
        confirm_count.append(1)
        return ConfirmResult.ALWAYS

    reg.register(ToolSchema("run", "运行", {"type": "object"}, "exec"), run_handler)
    ps = PermissionSystem(confirm=cb)
    llm = ScriptedLLM(
        [
            _chat_with_tools(ToolCall("c1", "run", "{}")),
            _chat_with_tools(ToolCall("c2", "run", "{}")),
            _chat_final("两次都执行了"),
        ]
    )
    ctx = SessionContext(session_id=sid, store=store, llm=llm, tools=reg, permissions=ps)

    run_turn("跑两次", ctx)
    assert executed == 2
    assert confirm_count == [1]  # 只问了一次
    store.close()


def test_run_turn_prunes_old_tool_results(tmp_path) -> None:
    """M3：低水位剪枝接入 run_turn——发送列表里旧工具结果被摘要，store 全量不动。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    for i in range(3):
        store.add_message(sid, Message("user", f"u{i}"))
        store.add_message(
            sid,
            Message("assistant", "", tool_calls=[ToolCall(f"c{i}", "shell", '{"command":"x"}')]),
        )
        content = "line\n" + "y" * 2000 if i == 0 else "ok"
        store.add_message(sid, Message("tool", content, tool_call_id=f"c{i}"))
    engine = ContextEngine(
        context_length=1000,
        cfg=ContextConfig(prune_percent=0.05, threshold_percent=0.9, tail_token_budget=100),
    )
    llm = FakeLLM()
    ctx = SessionContext(session_id=sid, store=store, llm=llm, engine=engine)
    run_turn("继续", ctx)

    sent = llm.calls[0]
    # 9 条历史 + 1 条新 user + 1 条规划提示（剪枝不改条数）
    assert len(sent) == 11
    assert any(m.role == "tool" and "\n" not in m.content and "已摘要" in m.content for m in sent)
    recent_tools = [m.content for m in sent if m.role == "tool"][-2:]
    assert all("\n" not in c and "已摘要" not in c for c in recent_tools)  # 尾部未动
    # store 全量历史保留（压缩只动发送列表）：9 历史 + 新 user + assistant 回复
    assert len(store.get_history(sid)) == 11
    store.close()


def test_run_turn_compresses_when_over_threshold(tmp_path) -> None:
    """M3：高水位压缩接入 run_turn——发送列表变短 + 标记消息，store 全量不动。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    for _ in range(20):
        store.add_message(sid, Message("user", "x" * 40))
        store.add_message(sid, Message("assistant", "y" * 40))
    engine = ContextEngine(
        context_length=1000,
        cfg=ContextConfig(prune_percent=0.01, threshold_percent=0.1, tail_token_budget=60),
    )
    llm = FakeLLM()
    ctx = SessionContext(session_id=sid, store=store, llm=llm, engine=engine)
    run_turn("再来", ctx)

    sent = llm.calls[0]
    assert len(sent) < 42  # 40 条历史 + user + 规划提示，压缩后更短
    assert any(m.role == "system" and "TailWindow" in m.content for m in sent)
    assert len(store.get_history(sid)) == 42  # 全量历史仍在库里
    store.close()


def test_first_message_sets_session_title(tmp_path) -> None:
    """M4：首条用户消息自动生成会话标题；后续消息不覆盖。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    llm = FakeLLM()
    ctx = SessionContext(session_id=sid, store=store, llm=llm)

    run_turn("帮我看看这个项目的结构", ctx)
    assert store.get_title(sid) == "帮我看看这个项目的结构"
    run_turn("继续", ctx)
    assert store.get_title(sid) == "帮我看看这个项目的结构"
    store.close()


def test_compacted_base_used_in_send_list(tmp_path) -> None:
    """M4：/compact 后的压缩列表作为发送底稿，store 全量历史不动。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    for i in range(3):
        store.add_message(sid, Message("user", f"u{i}"))
    engine = ContextEngine()
    engine.compacted = [Message("system", "【历史摘要（原 2 条）】要点"), Message("user", "u2")]
    llm = FakeLLM()
    ctx = SessionContext(session_id=sid, store=store, llm=llm, engine=engine)

    run_turn("继续", ctx)
    sent = llm.calls[0]
    assert sent[0].role == "system"  # M6：无计划时规划提示在最前
    assert "历史摘要" in sent[1].content  # 压缩底稿紧随其后
    assert sent[-1].content == "继续"
    # store 全量：3 条历史 + 新 user + assistant 回复（压缩底稿不落库）
    assert len(store.get_history(sid)) == 5
    store.close()


def test_max_tool_rounds_safety_valve(tmp_path, monkeypatch) -> None:
    """M5：超限后强制收尾——最后一次调用不带工具，消息全部落库、不崩溃。"""
    monkeypatch.setattr("vgent.agent.MAX_TOOL_ROUNDS", 3)
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    reg = ToolRegistry()
    executed = 0

    def run_handler(args: dict) -> str:
        nonlocal executed
        executed += 1
        return "ok"

    reg.register(ToolSchema("run", "运行", {"type": "object"}, "read"), run_handler)

    class LoopLLM:
        """每轮都调工具，直到 tools=None（收尾轮）。"""

        def __init__(self) -> None:
            self.calls: list[tuple[list | None, list[Message]]] = []

        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            self.calls.append((tools, list(messages)))
            if tools is None:
                return ChatResult(
                    messages=[Message("assistant", "收尾完成")], usage=Usage(1, 1, 2)
                )
            return _chat_with_tools(ToolCall("loop", "run", "{}"), content="")

    llm = LoopLLM()
    ctx = SessionContext(session_id=sid, store=store, llm=llm, tools=reg)

    result = run_turn("跑起来", ctx)
    assert result.messages[0].content == "收尾完成"
    # 3 轮工具 + 1 次无工具收尾
    assert len(llm.calls) == 4
    assert all(tools is not None for tools, _ in llm.calls[:3])
    assert llm.calls[3][0] is None
    assert executed == 3
    # store：user + 3*(assistant+tool) + 收尾 assistant = 8 条
    hist = store.get_history(sid)
    assert [m.role for m in hist] == ["user", "assistant", "tool", "assistant", "tool", "assistant", "tool", "assistant"]
    store.close()


# -- M6：任务计划（plan 消息化）与状态机 ---------------------------------------


def _plan_block(steps_json: str) -> str:
    return f"[vgent-plan]\n{steps_json}\n[/vgent-plan]"


def test_plan_generated_persisted_and_updated(tmp_path) -> None:
    """M6：模型输出计划 → 落库（历史只留一份）→ 状态变化后更新。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    reg = ToolRegistry()
    executed = 0

    def probe(args: dict) -> str:
        nonlocal executed
        executed += 1
        return "ok"

    reg.register(ToolSchema("probe", "探测", {"type": "object"}, "read"), probe)
    llm = ScriptedLLM(
        [
            _chat_with_tools(
                ToolCall("c1", "probe", "{}"),
                content=_plan_block(
                    '{"steps": [{"description": "扫描", "status": "pending"}, '
                    '{"description": "修改", "status": "pending"}]}'
                ),
            ),
            _chat_final(
                _plan_block(
                    '{"steps": [{"description": "扫描", "status": "done"}, '
                    '{"description": "修改", "status": "done"}]}'
                )
            ),
        ]
    )
    ctx = SessionContext(session_id=sid, store=store, llm=llm, tools=reg)
    run_turn("分析并修改", ctx)

    assert executed == 1
    assert ctx.plan is not None and len(ctx.plan.steps) == 2
    assert ctx.plan.steps[0].status == "done"  # 第二个响应更新了状态
    plans = [m for m in store.get_history(sid) if "[vgent-plan]" in m.content and m.role == "system"]
    assert len(plans) == 1  # 历史里只留最新一份
    assert "done" in plans[0].content
    # 首轮发送列表以规划提示开头（无计划时注入）
    assert llm.calls[0][0].role == "system"
    assert "[vgent-plan]" in llm.calls[0][0].content
    # 工具执行后：下一轮发送带「同步计划状态」轻推
    assert llm.calls[1][0].role == "system"
    assert "工具已执行完毕" in llm.calls[1][0].content
    store.close()


def test_plan_restored_on_resume(tmp_path) -> None:
    """M6：新会话进程从 SQLite 恢复计划（且不再注入规划提示）。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    llm = ScriptedLLM(
        [_chat_final(_plan_block('{"steps": [{"description": "a", "status": "pending"}]}'))]
    )
    ctx = SessionContext(session_id=sid, store=store, llm=llm)
    run_turn("任务", ctx)

    # 模拟重启：新上下文、同一 store（收尾会再标齐一次状态，需多一条响应）
    llm2 = ScriptedLLM(
        [
            _chat_final("继续"),
            _chat_final(_plan_block('{"steps": [{"description": "a", "status": "done"}]}')),
        ]
    )
    ctx2 = SessionContext(session_id=sid, store=store, llm=llm2)
    run_turn("继续", ctx2)
    assert ctx2.plan is not None and ctx2.plan.steps[0].description == "a"
    assert ctx2.plan.steps[0].status == "done"  # 收尾标齐
    assert llm2.calls[0][0].role == "user"  # 有计划 → 不注入提示
    store.close()


def test_bad_plan_json_falls_back(tmp_path) -> None:
    """M6：模型输出坏 JSON 计划 → 回退无计划模式，不落库不报错。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    llm = ScriptedLLM([_chat_final("[vgent-plan]\n{not json[/vgent-plan] 完成")])
    ctx = SessionContext(session_id=sid, store=store, llm=llm)
    run_turn("任务", ctx)
    assert ctx.plan is None
    assert all(
        not (m.role == "system" and "[vgent-plan]" in m.content)
        for m in store.get_history(sid)
    )
    store.close()


def test_plan_finalized_at_turn_end(tmp_path) -> None:
    """M6：回合结束模型未输出计划块 → 收尾调用把步骤状态标齐（done）。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    reg = ToolRegistry()

    def probe(args: dict) -> str:
        return "ok"

    reg.register(ToolSchema("probe", "探测", {"type": "object"}, "read"), probe)
    llm = ScriptedLLM(
        [
            _chat_with_tools(
                ToolCall("c1", "probe", "{}"),
                content=_plan_block('{"steps": [{"description": "扫描", "status": "pending"}]}'),
            ),
            _chat_final("完成（没有输出计划块）"),
            _chat_final(_plan_block('{"steps": [{"description": "扫描", "status": "done"}]}')),
        ]
    )
    ctx = SessionContext(session_id=sid, store=store, llm=llm, tools=reg)
    run_turn("任务", ctx)

    assert len(llm.calls) == 3  # 计划 + 执行 + 收尾标齐
    assert ctx.plan is not None and ctx.plan.steps[0].status == "done"
    plans = [m for m in store.get_history(sid) if "[vgent-plan]" in m.content and m.role == "system"]
    assert len(plans) == 1 and "done" in plans[0].content
    store.close()


def test_state_completed_and_failed(tmp_path) -> None:
    """M6：正常结束 → completed 落库；LLM 异常 → failed 落库。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()

    ctx = SessionContext(session_id=sid, store=store, llm=FakeLLM())
    run_turn("正常", ctx)
    assert ctx.state is AgentState.COMPLETED
    assert store.get_state(sid) == "completed"

    class BoomLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            raise RuntimeError("api down")

    ctx2 = SessionContext(session_id=sid, store=store, llm=BoomLLM())
    with pytest.raises(RuntimeError):
        run_turn("炸了", ctx2)
    assert ctx2.state is AgentState.FAILED
    assert store.get_state(sid) == "failed"
    store.close()
