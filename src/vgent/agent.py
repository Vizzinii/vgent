"""② Agent Loop 核心 —— 对话状态机（契约 v0.1 run_turn）。

M1：store + llm 接线。M2：接入 tools / permission，成为完整工具循环：
chat → tool_calls → 权限确认（契约③）→ 执行（契约⑤）→ 结果写回 → 再 chat，
直到模型不再调工具。M3：接入 ContextEngine（契约①②⑤）。
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from vgent.context import ContextEngine
from vgent.llm import ChatResult, LLMClient
from vgent.messages import Message, ToolCall
from vgent.permission import Approval, ConfirmResult, PermissionSystem
from vgent.store import SessionStore
from vgent.tools import Tool, ToolRegistry

MAX_TOOL_ROUNDS = 20  # 安全阀：单轮内最多工具往返次数，防失控循环（超限后强制收尾）


def _session_title(text: str) -> str:
    """会话标题：首条用户消息的首行，截断到 24 字符（与列表展示一致）。"""
    line = (text.strip().splitlines() or [""])[0].strip()
    if len(line) > 24:
        return line[:23] + "…"
    return line or "新会话"


@dataclass
class SessionContext:
    session_id: str
    store: SessionStore
    llm: LLMClient
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    permissions: PermissionSystem = field(default_factory=PermissionSystem)
    engine: ContextEngine = field(default_factory=ContextEngine)
    show_reasoning: bool = False  # M5：是否流式展示模型思考过程（/reasoning 切换）


def run_turn(
    user_input: str,
    ctx: SessionContext,
    on_delta: Callable[[str], None] | None = None,
    on_tool: Callable[[str, str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
) -> ChatResult:
    """跑一轮对话：历史 → 追加用户消息 → 工具循环 → 返回最终 ChatResult。

    on_delta：流式文本增量（契约④，CLI 渲染）；on_tool：每次工具执行后的
    (工具名, 结果摘要) 回调（CLI 显示状态行）；on_reasoning：思考过程分片（M5）。
    """
    msgs = ctx.store.get_history(ctx.session_id)
    if not msgs:
        # M4：首条用户消息自动生成会话标题（gemini-cli/openclaw 惯例）
        ctx.store.update_title(ctx.session_id, _session_title(user_input))
    # M4：/compact 后的压缩列表作为发送底稿（只影响发送列表，SQLite 全量历史不动）
    if ctx.engine.compacted is not None:
        msgs = list(ctx.engine.compacted)
    msgs.append(Message(role="user", content=user_input))
    ctx.store.add_message(ctx.session_id, msgs[-1])

    for _ in range(MAX_TOOL_ROUNDS):
        # M3 上下文引擎（只动发送列表，SQLite 全量历史不动）：
        msgs, _ = ctx.engine.prune_tool_results_only(msgs)  # 契约⑤ 低水位
        if ctx.engine.should_compress():  # 契约② 高水位
            msgs = ctx.engine.compress(msgs)
        # 传快照：LLM 客户端（或测试桩）可能持有该列表，后续 extend 不应污染它
        result = ctx.llm.chat(
            list(msgs), tools=ctx.tools.schemas(), on_delta=on_delta, on_reasoning=on_reasoning
        )
        ctx.engine.update_from_response(result.usage)  # 契约① usage 上报
        ctx.store.add_messages(ctx.session_id, result.messages)
        msgs.extend(result.messages)
        if not result.tool_calls:
            return result
        for tc in result.tool_calls:
            tool_msg = _dispatch_tool(tc, ctx, on_tool)
            ctx.store.add_message(ctx.session_id, tool_msg)
            msgs.append(tool_msg)

    # 超上限：最后一次调用不再给工具，让模型基于已有信息收尾
    final = ctx.llm.chat(list(msgs), tools=None, on_delta=on_delta, on_reasoning=on_reasoning)
    ctx.engine.update_from_response(final.usage)
    ctx.store.add_messages(ctx.session_id, final.messages)
    return final


def _dispatch_tool(
    tc: ToolCall, ctx: SessionContext, on_tool: Callable[[str, str], None] | None
) -> Message:
    """单个工具调用的完整生命周期，返回要写回的 tool 消息（契约⑤）。"""
    tool: Tool | None = ctx.tools.get(tc.name)
    if tool is None:
        return Message("tool", f"未知工具：{tc.name}（请只使用已提供的工具）", tool_call_id=tc.id)
    args, err = _safe_parse(tc.arguments)
    if err:
        # 决策 9：坏参数不崩溃，把解析错误回喂模型修正
        return Message("tool", f"参数解析失败：{err}", tool_call_id=tc.id)

    approval = ctx.permissions.check(tool.schema, args)
    if approval is Approval.NEED_CONFIRM:
        result = ctx.permissions.confirm(tool.schema, args)
        if result is ConfirmResult.REJECT:
            return Message("tool", f"用户拒绝了工具 {tc.name} 的调用", tool_call_id=tc.id)
    elif approval is Approval.DENIED:
        return Message("tool", f"工具 {tc.name} 被权限系统拒绝", tool_call_id=tc.id)

    try:
        out = ctx.tools.execute(tc.name, args)
    except Exception as exc:  # noqa: BLE001 — 工具异常也要回喂模型（决策 9：容忍并自纠正）
        out = f"工具 {tc.name} 执行出错：{exc}"
    if on_tool:
        on_tool(tc.name, out)
    return Message("tool", out, tool_call_id=tc.id)


def _safe_parse(arguments: str) -> tuple[dict | None, str | None]:
    """容忍坏参数（决策 9）：返回 (解析结果, 错误)；错误时结果为空。"""
    try:
        data = json.loads(arguments) if arguments and arguments.strip() else {}
    except json.JSONDecodeError as exc:
        return None, f"JSON 非法：{exc}"
    if not isinstance(data, dict):
        return None, f"应为 JSON 对象，实际为 {type(data).__name__}"
    return data, None
