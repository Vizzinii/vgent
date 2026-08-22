"""② Agent Loop 核心 —— 对话状态机（契约 v0.1 run_turn）。

M1：store + llm 接线。M2：接入 tools / permission，成为完整工具循环：
chat → tool_calls → 权限确认（契约③）→ 执行（契约⑤）→ 结果写回 → 再 chat，
直到模型不再调工具。M3：接入 ContextEngine（契约①②⑤）。
M6：任务计划（plan 消息化）+ 显式 AgentState——每轮解析模型输出的
[vgent-plan] 块并落库，状态转场随轮次持久化。
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from vgent.context import ContextEngine
from vgent.llm import ChatResult, LLMClient
from vgent.memory.episodic import EpisodicMemory, current_project, memory_note_text, summarize
from vgent.memory.pipeline import MemoryPipeline, should_extract, slice_round
from vgent.memory.store import MemoryFileStore
from vgent.messages import Message, ToolCall
from vgent.permission import Approval, ConfirmResult, PermissionSystem
from vgent.reflection import MAX_REFLECT_ROUNDS, looks_failed, reflect
from vgent.snapshot import SnapshotStore
from vgent.state import AgentState
from vgent.store import SessionStore
from vgent.task import PLAN_HINT, TaskPlan, plan_from_messages
from vgent.tools import Tool, ToolRegistry

MAX_TOOL_ROUNDS = 20  # 安全阀：单轮内最多工具往返次数，防失控循环（超限后强制收尾）


def _session_title(text: str) -> str:
    """会话标题：首条用户消息的首行，截断到 24 字符（与列表展示一致）。"""
    line = (text.strip().splitlines() or [""])[0].strip()
    if len(line) > 24:
        return line[:23] + "…"
    return line or "新会话"


def _send_with_anchors(
    msgs: list[Message],
    *,
    cwd_anchor: Message | None,
    instruction_anchors: list[Message],
    hint: Message | None,
    plan_nudge: Message | None,
    reflection_note: Message | None,
    memory_notes: list[Message] | None,
    memory_summary: Message | None = None,
) -> list[Message]:
    """M12：组装发送列表（锚点顺序与原 inline 逻辑一致，供估算触发压缩后重建复用）。

    memory_summary：M12-C 项目级记忆总览（每轮注入，非一次性；None 时跳过）。
    """
    send = list(msgs)
    if memory_summary is not None:
        send.insert(0, memory_summary)
    if cwd_anchor is not None:
        send.insert(0, cwd_anchor)
    if instruction_anchors:
        send = instruction_anchors + send
    if hint is not None:
        send.insert(0, hint)
    if plan_nudge is not None:
        send.insert(0, plan_nudge)
    if reflection_note is not None:
        send.insert(0, reflection_note)
    if memory_notes is not None:
        send = memory_notes + send
    return send


def _tools_schema_json(registry: ToolRegistry) -> str | None:
    """M12：tools schema 的固定开销文本（估算时计入；无工具则 None）。"""
    schemas = registry.schemas()
    if not schemas:
        return None
    return json.dumps(schemas, ensure_ascii=False)


def _persist_compacted(
    ctx: SessionContext, before: list[Message], after: list[Message]
) -> None:
    """M12：压缩实际发生时把摘要 + 保留尾部 + 边界落库，供恢复会话重建发送底稿。

    after = [头部, 摘要/标记消息, *保留尾部]；边界 = 压缩时刻的最后消息 id，
    之后新增的消息由 get_history_after 续接（尾部消息与 messages 表同源，不重复）。
    """
    if after is before or len(after) < 2:
        return
    marker = after[1]
    if marker.role != "system" or not marker.content:
        return
    boundary = ctx.store.last_message_id(ctx.session_id)
    if boundary is None:
        return
    ctx.store.upsert_compact(ctx.session_id, marker.content, list(after[2:]), boundary)


def _compacted_from_store(ctx: SessionContext) -> list[Message] | None:
    """M12：从 SQLite 压缩记录重建发送底稿（恢复会话后不发全量历史）。

    底稿 = [头部, 摘要, *保留尾部, *边界之后的新消息]——与压缩时刻的内存视图一致。
    """
    comp = ctx.store.get_compact(ctx.session_id)
    if comp is None:
        return None
    summary, retained, boundary = comp
    history = ctx.store.get_history(ctx.session_id)
    after = ctx.store.get_history_after(ctx.session_id, boundary)
    base: list[Message] = [Message("system", summary), *retained, *after]
    if history:
        base.insert(0, history[0])
    return base


@dataclass
class SessionContext:
    session_id: str
    store: SessionStore
    llm: LLMClient
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    permissions: PermissionSystem = field(default_factory=PermissionSystem)
    engine: ContextEngine = field(default_factory=ContextEngine)
    show_reasoning: bool = False  # M5：是否流式展示模型思考过程（/reasoning 切换）
    plan: TaskPlan | None = None  # M6：当前任务计划（恢复自历史/每轮更新）
    state: AgentState = AgentState.IDLE  # M6：当前状态（每轮结束落库）
    memory: EpisodicMemory | None = None  # M8：episodic 记忆（跨会话任务摘要）
    memory_auto: bool = False  # M8：任务计划完成时自动存摘要（每会话一次）
    mcp_tools: dict[str, list[str]] = field(default_factory=dict)  # M9：已加载的 MCP 工具 {server: [names]}
    instructions: str | None = None  # M10：项目指令内容（AGENTS.md/CLAUDE.md）
    instructions_name: str | None = None  # M10：指令来源文件名（cli 解析时记录）
    user_instructions: str | None = None  # P6：用户级指令（~/.vgent/AGENTS.md）
    ext_commands: dict[str, Callable] = field(default_factory=dict)  # M10：外部命令 {name: run(ctx, args)}
    data_dir: Path | None = None  # P2：/allow 持久化到 config.toml 用（cli/web 注入）
    snapshots: SnapshotStore | None = None  # M12-B：快照/恢复（cli/web 注入；None 时全路径 no-op）
    memory_file_store: MemoryFileStore | None = None  # M12-C：记忆文件存储（/memory 命令用）
    memory_pipeline: MemoryPipeline | None = None  # M12-C：自动两阶段记忆管线（memory_auto 时注入）
    memory_summary: str | None = None  # M12-C：项目级记忆总览（MemoryFileStore.read_summary，每轮注入）


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
    M6：无计划时注入规划提示（不落库），解析模型输出的 [vgent-plan] 块并落库；
    状态转场（PLANNING/EXECUTING/WAITING_PERMISSION/COMPLETED/FAILED）随轮次持久化。
    """
    msgs = ctx.store.get_history(ctx.session_id)
    first_turn = not msgs
    if not msgs:
        # M4：首条用户消息自动生成会话标题（gemini-cli/openclaw 惯例）
        ctx.store.update_title(ctx.session_id, _session_title(user_input))
    # M4：/compact 后的压缩列表作为发送底稿（只影响发送列表，SQLite 全量历史不动）
    # M12：进程内无底稿时从 SQLite 压缩记录重建（恢复会话后不发全量历史）
    # 底稿一次一清（评审 F1）：消费后置 None，下轮走 _compacted_from_store 重建
    # （含压缩边界之后的新消息）——否则同进程 /compact 后续回合永远拿旧快照，丢中间轮。
    if ctx.engine.compacted is not None:
        msgs = list(ctx.engine.compacted)
        ctx.engine.compacted = None
    elif (rebuilt := _compacted_from_store(ctx)) is not None:
        msgs = list(rebuilt)
    # M12 中断修复：Ctrl-C 落在「assistant(tool_calls) 已落库、工具结果未落库」的窗口
    # 会在库里留下孤儿 tool_calls——发送前清洗（SQLite 全量历史不动，只修发送列表），
    # 防止恢复会话后 API 因「tool_calls 无对应 tool 响应」返回 400。
    msgs = ContextEngine._cleanup_tool_pairs(msgs)
    msgs.append(Message(role="user", content=user_input))
    ctx.store.add_message(ctx.session_id, msgs[-1])

    # M6：恢复计划 + 初始化状态（无计划 → PLANNING；有 → EXECUTING）
    ctx.plan = plan_from_messages(msgs)
    ctx.state = AgentState.EXECUTING if ctx.plan else AgentState.PLANNING
    hint: Message | None = None if ctx.plan else Message("system", PLAN_HINT)
    plan_nudge: Message | None = None  # 工具执行后轻推模型同步计划状态
    # M7：失败反思——本轮失败计数与待注入的反思消息（上限内失败才反思）
    reflections = 0
    reflection_note: Message | None = None
    # 工作目录锚点：新会话首轮注入一次，防止模型在错误 CWD 下瞎找（真机首跑踩坑）
    cwd_anchor: Message | None = (
        Message("system", f"工作目录：{os.getcwd()}；工具的相对路径基于此。") if first_turn else None
    )
    # P6/M10：用户级（~/.vgent/AGENTS.md）+ 项目级指令——新会话首轮注入一次，不落库
    # （与 cwd_anchor 同模式）；顺序：用户在前、项目在后
    instruction_anchors: list[Message] = []
    if first_turn and ctx.user_instructions:
        instruction_anchors.append(
            Message("system", f"用户指令（用户级）：\n{ctx.user_instructions}")
        )
    if first_turn and ctx.instructions:
        instruction_anchors.append(
            Message(
                "system",
                f"项目指令（{ctx.instructions_name or 'AGENTS.md'}）：\n{ctx.instructions}",
            )
        )
    # M8：自动回忆——用户消息命中已存记忆主题 → 注入 [记忆]（不落库，一次性）
    # P5：只搜当前项目（防跨项目串味）；M12-C：>1 天条目附新鲜度警告
    memory_notes: list[Message] | None = None
    if ctx.memory is not None:
        hits = [
            e for e in ctx.memory.search(user_input, limit=2, project=current_project())
            if not _memory_already_present(msgs, e.topic)
        ]
        if hits:
            memory_notes = [Message("system", memory_note_text(e)) for e in hits]

    # M12-C：项目级记忆总览（memory_summary.md 截断），每轮注入（非一次性）
    # 每轮从文件重读（评审 F4）：管线 stage2 会重写 summary，启动时读的快照会陈旧；
    # 无 memory_file_store 时退回 ctx.memory_summary 直填（测试桩口径，R4）
    if ctx.memory_file_store is not None:
        summary = ctx.memory_file_store.read_summary()
        ctx.memory_summary = (
            None if ctx.memory_file_store.summary_is_placeholder() else summary
        )
    memory_summary_note: Message | None = None
    if ctx.memory_summary:
        memory_summary_note = Message("system", f"[记忆总览]\n{ctx.memory_summary}")

    # M12：发送前估算用的模型名（测试桩 FakeLLM 无 cfg 时回退 None → cl100k_base）
    _provider = getattr(getattr(ctx.llm, "cfg", None), "provider", None)
    llm_model = getattr(_provider, "model", None) if _provider is not None else None

    # M12-B：回合快照（写盘前登记、回合末封存；快照是辅助功能，失败静默不阻断对话）
    if ctx.snapshots is not None:
        try:
            ctx.snapshots.begin_turn()
        except Exception:  # noqa: BLE001, S110
            pass

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            # M3 上下文引擎（只动发送列表，SQLite 全量历史不动）：
            msgs, _ = ctx.engine.prune_tool_results_only(msgs)  # 契约⑤ 低水位
            if ctx.engine.should_compress():  # 契约② 高水位（usage 校准）
                before = msgs
                msgs = ctx.engine.compress(msgs)
                _persist_compacted(ctx, before, msgs)  # M12：压缩结果落库
            send = _send_with_anchors(
                msgs,
                cwd_anchor=cwd_anchor,
                instruction_anchors=instruction_anchors,
                hint=hint,
                plan_nudge=plan_nudge,
                reflection_note=reflection_note,
                memory_notes=memory_notes,
                memory_summary=memory_summary_note,
            )
            # M12：发送前 tiktoken 精确估算（含 tools schema 固定开销 + 预留输出），
            # 与 usage 校准并列——弥补「只按 API usage 触发」看不到的 system/tools 开销
            if ctx.engine.should_compress_estimated(
                send, model=llm_model, fixed_extra=_tools_schema_json(ctx.tools)
            ):
                before = msgs
                msgs = ctx.engine.compress(msgs)
                _persist_compacted(ctx, before, msgs)
                send = _send_with_anchors(
                    msgs,
                    cwd_anchor=cwd_anchor,
                    instruction_anchors=instruction_anchors,
                    hint=hint,
                    plan_nudge=plan_nudge,
                    reflection_note=reflection_note,
                    memory_notes=memory_notes,
                    memory_summary=memory_summary_note,
                )
            # 消费一次性锚点（首轮注入后失效；hint/plan_nudge 每轮重置语义不变）
            cwd_anchor = None
            instruction_anchors = []
            reflection_note = None
            memory_notes = None
            # 传快照：LLM 客户端（或测试桩）可能持有该列表，后续 extend 不应污染它
            result = ctx.llm.chat(
                send, tools=ctx.tools.schemas(), on_delta=on_delta, on_reasoning=on_reasoning
            )
            ctx.engine.update_from_response(result.usage)  # 契约① usage 上报
            ctx.store.add_messages(ctx.session_id, result.messages)
            msgs.extend(result.messages)
            # M6：解析模型输出的计划块并持久化（坏 JSON / 模型未输出 → 回退无计划模式）
            plan = plan_from_messages(result.messages)
            if plan is not None and plan != ctx.plan:
                ctx.plan = plan
                ctx.store.upsert_plan_message(ctx.session_id, plan.plan_message().content)
                hint = None
                plan_nudge = None
            if not result.tool_calls:
                # M6：计划存在但最终状态未标齐（无计划块 / 仍有 pending）→ 收尾标齐
                if ctx.plan is not None and (plan is None or not ctx.plan.done):
                    _finalize_plan(ctx, msgs)
                ctx.state = AgentState.COMPLETED
                _persist_state(ctx)
                _maybe_auto_memory(ctx, msgs)  # M8：计划完成 → 自动存摘要（可配置）
                _maybe_submit_memory_round(ctx, msgs)  # M12-C：本轮切片入管线（非阻塞）
                return result
            failed_any = False
            for tc in result.tool_calls:
                tool_msg = _dispatch_tool(tc, ctx, on_tool)
                ctx.store.add_message(ctx.session_id, tool_msg)
                msgs.append(tool_msg)
                if looks_failed(tool_msg.content):
                    failed_any = True
            # M7：失败 → 显式反思一次（上限内），结果注入下一轮发送列表（不落库，
            # 一次性引导；模型不配合/反思失败自动回退决策 9 的错误回喂）
            if failed_any and reflections < MAX_REFLECT_ROUNDS:
                reflections += 1
                note = reflect(msgs, ctx.llm)
                if note:
                    reflection_note = Message("system", f"[反思] {note}")
            if ctx.plan is not None:
                plan_nudge = Message(
                    "system",
                    "工具已执行完毕。若计划中有步骤已完成或失败，"
                    "请输出更新后的完整 [vgent-plan] 块（无变化则不必输出）。",
                )

        # 超上限：最后一次调用不再给工具，让模型基于已有信息收尾
        final = ctx.llm.chat(list(msgs), tools=None, on_delta=on_delta, on_reasoning=on_reasoning)
        ctx.engine.update_from_response(final.usage)
        ctx.store.add_messages(ctx.session_id, final.messages)
        ctx.state = AgentState.COMPLETED
        _persist_state(ctx)
        _maybe_submit_memory_round(ctx, msgs)  # M12-C：超限收尾也算完整一轮
        return final
    except Exception:
        ctx.state = AgentState.FAILED
        _persist_state(ctx)
        raise
    finally:
        # M12-B：回合末封存快照（异常路径也封存；失败静默不覆盖原异常）
        if ctx.snapshots is not None:
            try:
                ctx.snapshots.seal_turn()
            except Exception:  # noqa: BLE001, S110
                pass


def _finalize_plan(ctx: SessionContext, msgs: list[Message]) -> None:
    """M6：回合结束且模型未更新计划 → 追加一次调用把步骤状态标齐。

    结果不落库（计划已作为 system 消息持久化），失败不阻断对话。
    """
    prompt = Message(
        "system",
        "所有工具已执行完毕。请在回复正文中直接输出最终更新的 [vgent-plan] 块"
        "（不要放在思考内容里）：已完成步骤标为 done、失败标为 failed、"
        "未执行的保持 pending。只输出计划块。",
    )
    try:
        result = ctx.llm.chat([prompt, *msgs])
    except Exception:  # noqa: BLE001 — 计划收尾失败不影响对话
        return
    ctx.engine.update_from_response(result.usage)
    final_plan = plan_from_messages(result.messages)
    if final_plan is not None and final_plan != ctx.plan:
        ctx.plan = final_plan
        ctx.store.upsert_plan_message(ctx.session_id, final_plan.plan_message().content)


def _persist_state(ctx: SessionContext) -> None:
    """M6：把当前 Agent 状态落库（供恢复/展示）。"""
    ctx.store.set_state(ctx.session_id, ctx.state.value)


def _memory_already_present(msgs: list[Message], topic: str) -> bool:
    """历史里是否已有该主题的 [记忆]（防自动回忆重复注入）。"""
    return any(
        m.role == "system" and m.content.startswith("[记忆]") and topic in m.content
        for m in msgs
    )


def _maybe_auto_memory(ctx: SessionContext, msgs: list[Message]) -> None:
    """M8：memory_auto + 任务计划完成 → 自动生成并存储会话摘要（每会话一次）。"""
    if not (ctx.memory and ctx.memory_auto):
        return
    if ctx.plan is None or not ctx.plan.done:
        return
    if ctx.memory.has_session(ctx.session_id):
        return
    title = ctx.store.get_title(ctx.session_id) or (
        ctx.plan.steps[0].description if ctx.plan.steps else "会话"
    )
    summary = summarize(msgs, ctx.llm, title)
    if summary:
        ctx.memory.add(title, summary, ctx.session_id, title)


def _maybe_submit_memory_round(ctx: SessionContext, msgs: list[Message]) -> None:
    """M12-C：本轮对话切片提交给记忆管线（非阻塞、不写 store、不调 ctx.llm，R3）。

    管线未注入（默认 None）或本轮不值得抽取时全路径 no-op——现有 run_turn
    测试的 llm.calls 精确断言与消息序列断言不受影响。
    """
    if ctx.memory_pipeline is None:
        return
    rc = slice_round(msgs, workspace=Path(os.getcwd()), session_id=ctx.session_id)
    if should_extract(rc):
        ctx.memory_pipeline.submit(rc)


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
        ctx.state = AgentState.WAITING_PERMISSION  # M6：确认交互中
        result = ctx.permissions.confirm(tool.schema, args)
        ctx.state = AgentState.EXECUTING
        if result is ConfirmResult.REJECT:
            return Message("tool", f"用户拒绝了工具 {tc.name} 的调用", tool_call_id=tc.id)
    elif approval is Approval.DENIED:
        return Message("tool", f"工具 {tc.name} 被权限系统拒绝", tool_call_id=tc.id)

    _note_snapshot_before_write(ctx, tc.name, args)  # M12-B：写盘前登记快照

    try:
        out = ctx.tools.execute(tc.name, args)
    except Exception as exc:  # noqa: BLE001 — 工具异常也要回喂模型（决策 9：容忍并自纠正）
        out = f"工具 {tc.name} 执行出错：{exc}"
    if on_tool:
        on_tool(tc.name, out)
    return Message("tool", out, tool_call_id=tc.id)


def _note_snapshot_before_write(ctx: SessionContext, tool_name: str, args: dict) -> None:
    """M12-B：write/edit 改盘前登记原文快照（相对 cwd posix；越界/失败跳过不阻断）。

    放在 agent 派发层而非工具 handler 内——工具 handler 契约保持纯 `handler(args)`（R2）。
    路径解析与 tools.py 同规则（相对路径基于 os.getcwd()）。
    """
    if ctx.snapshots is None or tool_name not in ("write_file", "edit_file"):
        return
    raw = str(args.get("path", "")).strip()
    if not raw:
        return
    try:
        p = Path(raw)
        resolved = p.resolve(strict=False) if p.is_absolute() else (
            Path(os.getcwd()) / p
        ).resolve(strict=False)
        rel = resolved.relative_to(Path(os.getcwd()).resolve()).as_posix()
    except (OSError, ValueError):
        return  # 越界/坏路径：快照不跟踪，工具照常执行
    try:
        ctx.snapshots.note_before_write(rel)
    except Exception:  # noqa: BLE001, S110 — 快照失败不影响工具执行
        pass


def _safe_parse(arguments: str) -> tuple[dict | None, str | None]:
    """容忍坏参数（决策 9）：返回 (解析结果, 错误)；错误时结果为空。"""
    try:
        data = json.loads(arguments) if arguments and arguments.strip() else {}
    except json.JSONDecodeError as exc:
        return None, f"JSON 非法：{exc}"
    if not isinstance(data, dict):
        return None, f"应为 JSON 对象，实际为 {type(data).__name__}"
    return data, None
