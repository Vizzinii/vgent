"""① CLI/REPL 界面层。

M2：最小闭环 + 工具/权限确认交互。
M4（2026-08-20）：/compact（Summarize LLM 摘要）+ 状态栏（token 用量）+ 会话 title 自动生成；
zcode 化前置：--provider/--new/--resume/--list-sessions/--delete-session/--version
flag 族 + /list /delete 命令 + 记住上次会话（~/.vgent/last_session）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markup import escape

from vgent import __version__
from vgent.agent import SessionContext, run_turn
from vgent.config import Config, load_config
from vgent.context import ContextEngine
from vgent.llm import LLMClient
from vgent.messages import Message
from vgent.permission import ConfirmResult, PermissionSystem
from vgent.store import SessionStore
from vgent.task import plan_from_messages
from vgent.tools import ToolSchema, default_tools

HELP = """命令：
  /new            新建会话
  /resume         列出并恢复会话
  /list           列出会话
  /delete         删除会话（按编号，当前会话不可删）
  /compact        压缩当前会话（LLM 摘要中间历史，下次对话生效）
  /plan           查看任务计划（/plan new 清除并重新规划）
  /reasoning      切换思考过程展示（开/关，默认关）
  /help           显示帮助
  /exit           退出
"""


def setup_logging(level: str) -> None:
    """轻量日志：stdlib logging，由各模块直接使用（决策：不设独立组件）。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx/httpx2 的请求行（401 等）属于诊断级噪音，别污染 REPL 输出
    # （openai SDK 3.x 的日志器名是 httpx2，不是 httpx）
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)


def _make_prompter() -> Callable[[str], str]:
    """返回 prompt(ps) -> str。

    优先 prompt_toolkit（真终端下有多行编辑/历史/补全）；在 Git Bash/mintty、管道输入等
    拿不到 Windows 控制台的场景自动退回 input()（Windows 已知坑：NoConsoleScreenBufferError，
    mintty 下 TERM=xterm-256color 会让 Win32Output 构造失败）。
    """
    try:
        session = PromptSession(history=InMemoryHistory())
        return session.prompt
    except Exception:  # noqa: BLE001 — 构造失败即退回基础输入，任何环境都能跑
        return lambda ps: input(ps)


def _print_sessions(
    store: SessionStore, console: Console, current: str | None = None
) -> None:
    """列出会话；current 会话标 *。"""
    sessions = store.list_sessions()
    if not sessions:
        console.print("[yellow]暂无历史会话。[/yellow]")
        return
    for i, s in enumerate(sessions, 1):
        mark = " *" if s.id == current else ""
        # markup=False：会话标题来自用户消息，可能含 []，不当样式解析
        console.print(
            f"  {i}. {s.title[:24]}  ({s.message_count} 条, {s.created_at[:16]}){mark}",
            markup=False,
        )


def _pick_session(
    store: SessionStore,
    console: Console,
    prompt: Callable[[str], str],
    last_sid: str | None = None,
) -> str | None:
    """启动时选会话：数字=恢复，0=上次会话，Enter=新建，q=退出。返回 session_id 或 None(新建)。"""
    sessions = store.list_sessions()
    if sessions:
        console.print("[bold]已有会话：[/bold]")
        if last_sid:
            meta = next((s for s in sessions if s.id == last_sid), None)
            if meta:
                console.print(
                    f"  0. [bold]上次会话[/bold] ({escape(meta.title[:24])}, {meta.message_count} 条)"
                )
        _print_sessions(store, console)
    console.print("[dim]输入编号恢复会话；0=上次会话；直接回车新建；q 退出。[/dim]")
    while True:
        try:
            choice = prompt("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return "QUIT"
        if choice == "":
            return None
        if choice.lower() in ("q", "quit"):
            return "QUIT"
        if choice == "0" and last_sid:
            return last_sid
        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(sessions):
                return sessions[n - 1].id
        console.print("[red]无效输入，重试。[/red]")


def _make_confirm(
    console: Console, prompt: Callable[[str], str]
) -> Callable[[ToolSchema, dict], ConfirmResult]:
    """权限确认交互（契约③的 confirm）：y 一次 / a 本会话总是 / n 拒绝（rich + 输入适配）。"""

    def confirm(tool: ToolSchema, args: dict) -> ConfirmResult:
        console.print(f"\n[yellow]工具调用需确认：[/yellow][bold]{tool.name}[/bold]（{tool.permission} 档）")
        console.print(f"  参数：[dim]{escape(str(args))}[/dim]")
        while True:
            try:
                choice = prompt("[y]执行一次  [a]本会话总是允许  [n]拒绝 > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return ConfirmResult.REJECT
            if choice in ("y", "yes"):
                return ConfirmResult.APPROVE
            if choice in ("a", "always", "all"):
                return ConfirmResult.ALWAYS
            if choice in ("n", "no", ""):
                return ConfirmResult.REJECT
            console.print("[red]输入 y / a / n[/red]")

    return confirm


# -- 启动 flag 族（zcode / gemini-cli 风格） --------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vgent", description="通用 agent CLI")
    parser.add_argument("--provider", help="使用哪个 provider（config.toml 的 [providers] 名）")
    parser.add_argument("--version", action="version", version=f"vgent {__version__}")
    parser.add_argument("--new", action="store_true", help="跳过会话选择，直接新建会话")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        metavar="ID|N|last",
        help="恢复会话：缺省=上次会话；N=列表编号；或直接给会话 id",
    )
    parser.add_argument("--list-sessions", action="store_true", help="列出会话后退出")
    parser.add_argument("--delete-session", metavar="ID", help="删除指定会话后退出")
    return parser


def _last_session_path(cfg: Config) -> Path:
    return cfg.data_dir / "last_session"


def _remember_session(path: Path, sid: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sid, encoding="utf-8")


def _last_session(store: SessionStore, path: Path) -> str | None:
    """上次会话 id；会话已被删除则视为不存在。"""
    if not path.exists():
        return None
    sid = path.read_text(encoding="utf-8").strip()
    return sid if sid and store.get_title(sid) else None


def _resolve_resume(store: SessionStore, arg: str, last_path: Path) -> str | None:
    """--resume 参数解析：last / 编号 / 会话 id。"""
    sessions = store.list_sessions()
    if arg == "last":
        sid = _last_session(store, last_path)
        return sid or (sessions[0].id if sessions else None)
    if arg.isdigit():
        n = int(arg)
        if 1 <= n <= len(sessions):
            return sessions[n - 1].id
        return None
    return arg if store.get_title(arg) else None


def _make_summarizer(llm: LLMClient) -> Callable[[list[Message]], str]:
    """Summarize 策略的 LLM 摘要器：把中间段历史压成几句要点（/compact 用）。"""

    def summarize(middle: list[Message]) -> str:
        prompt = Message(
            "system",
            "你是会话压缩器。把下面的对话历史压缩成 3~5 句要点摘要，"
            "保留关键事实、已做的决定和未完成的任务；只输出摘要本身。",
        )
        result = llm.chat([prompt, *middle])
        return (result.messages[0].content or "").strip()

    return summarize


def _headless(store: SessionStore, args: argparse.Namespace, console: Console) -> int | None:
    """一次性 flag 动作（--list-sessions / --delete-session）；未命中返回 None 进入 REPL。"""
    if args.list_sessions:
        _print_sessions(store, console)
        return 0
    if args.delete_session:
        sid = args.delete_session
        if not store.get_title(sid):
            console.print(f"[red]会话不存在：{sid}[/red]")
            return 1
        store.delete_session(sid)
        console.print(f"[dim]已删除会话 {sid[:8]}[/dim]")
        return 0
    return None


def _resolve_start_session(
    store: SessionStore,
    args: argparse.Namespace,
    prompt: Callable[[str], str],
    console: Console,
    last_path: Path,
) -> str:
    """决定启动进入哪个会话；返回 session_id（'QUIT' 表示退出）。"""
    if args.new:
        return store.create_session()
    if args.resume is not None:
        sid = _resolve_resume(store, args.resume, last_path)
        if sid is None:
            console.print("[red]没有可恢复的会话（--resume）。[/red]")
            return "QUIT"
        return sid
    picked = _pick_session(store, console, prompt, _last_session(store, last_path))
    if picked == "QUIT":
        return "QUIT"
    return picked or store.create_session()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(provider=args.provider)
    setup_logging(cfg.log_level)
    console = Console()

    db_path = cfg.data_dir / "sessions" / "vgent.db"
    store = SessionStore(db_path)
    try:
        code = _headless(store, args, console)
        if code is not None:
            return code
        console.print(
            f"[bold green]vgent[/bold green] v{__version__} — \\[{cfg.provider.name}] {cfg.provider.model}"
        )
        if not cfg.api_key_resolved():
            hint = "编辑 ~/.vgent/config.toml 补 api_key"
            if cfg.provider.api_key_env:
                hint += f" 或设置环境变量 {cfg.provider.api_key_env}"
            console.print(f"[yellow]未配置 API key[/yellow]：{hint}（对话将报错）。")
        prompt = _make_prompter()
        last_path = _last_session_path(cfg)
        session_id = _resolve_start_session(store, args, prompt, console, last_path)
        if session_id == "QUIT":
            return 0
        _remember_session(last_path, session_id)
        llm = LLMClient(cfg)
        tools = default_tools()
        permissions = PermissionSystem(confirm=_make_confirm(console, prompt))
        engine = ContextEngine(cfg.provider.context_length, cfg.context)
        engine.summarizer = _make_summarizer(llm)  # M4：/compact 的 LLM 摘要器
        ctx = SessionContext(
            session_id=session_id,
            store=store,
            llm=llm,
            tools=tools,
            permissions=permissions,
            engine=engine,
            show_reasoning=cfg.show_reasoning,
        )
        _repl(ctx, console, prompt, last_path)
    finally:
        store.close()
    return 0


def _on_tool(console: Console) -> Callable[[str, str], None]:
    """工具执行后的状态行（契约⑤的可视化：让用户看到 agent 在做什么）。"""

    def show(name: str, out: str) -> None:
        first = escape(out.splitlines()[0][:60]) if out else ""
        console.print(f"\n[dim]  → {name}: {first}[/dim]")

    return show


def _repl(
    ctx: SessionContext,
    console: Console,
    prompt: Callable[[str], str],
    last_path: Path,
) -> None:
    console.print(f"[dim]会话 {ctx.session_id[:8]} 开始（/help 查看命令）[/dim]")
    session_tokens = 0  # M4 状态栏：本会话累计 token（切换会话时清零）
    while True:
        try:
            text = prompt("你> ")
        except (KeyboardInterrupt, EOFError):
            console.print()
            break
        text = text.strip()
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break
        if text == "/help":
            console.print(HELP)
            continue
        if text == "/new":
            ctx.session_id = ctx.store.create_session()
            _remember_session(last_path, ctx.session_id)
            ctx.engine.compacted = None
            session_tokens = 0
            console.print(f"[dim]已新建会话 {ctx.session_id[:8]}[/dim]")
            continue
        if text in ("/resume", "/list", "/list-sessions"):
            _resume_inline(ctx, console, prompt, last_path, action=text)
            if text == "/resume":  # 切换会话：清掉压缩底稿与累计
                ctx.engine.compacted = None
                session_tokens = 0
            continue
        if text in ("/delete", "/delete-session"):
            _delete_inline(ctx, console, prompt)
            continue
        if text == "/compact":
            _compact_inline(ctx, console)
            continue
        if text == "/plan" or text.startswith("/plan "):
            _plan_inline(ctx, console, redo=("new" in text or "redo" in text))
            continue
        if text == "/reasoning":
            ctx.show_reasoning = not ctx.show_reasoning
            console.print(f"[dim]思考过程展示：{'开' if ctx.show_reasoning else '关'}[/dim]")
            continue
        try:
            console.print("[cyan]assistant>[/cyan] ", end="")
            # M5：开启时以 dim 样式流式渲染模型思考过程（与正文颜色区分）
            on_reasoning = (
                (lambda d: console.print(d, end="", style="dim", markup=False))
                if ctx.show_reasoning
                else None
            )
            result = run_turn(
                text,
                ctx,
                on_delta=lambda d: console.print(d, end="", markup=False),
                on_tool=_on_tool(console),
                on_reasoning=on_reasoning,
            )
            console.print()
            if result.usage:  # M4 状态栏：token 用量（M6 加计划进度与状态）
                session_tokens += result.usage.total_tokens
                parts = [
                    (
                        f"tok ↑{result.usage.prompt_tokens} ↓{result.usage.completion_tokens} "
                        f"= {result.usage.total_tokens}"
                    ),
                    f"会话累计 {session_tokens}",
                ]
                if ctx.engine.compression_count:
                    parts.append(f"压缩 {ctx.engine.compression_count} 次")
                if ctx.plan and ctx.plan.steps:
                    done = sum(1 for s in ctx.plan.steps if s.status == "done")
                    parts.append(f"计划 {done}/{len(ctx.plan.steps)}")
                parts.append(f"状态 {ctx.state.value}")
                console.print("[dim]  " + "；".join(parts) + "[/dim]")
        except Exception as exc:  # noqa: BLE001 — REPL 顶层兜底：任何错误都回到输入，不崩溃
            console.print(f"\n[red]调用失败：{exc}[/red]")


def _resume_inline(
    ctx: SessionContext,
    console: Console,
    prompt: Callable[[str], str],
    last_path: Path,
    action: str,
) -> None:
    """/resume（列+切换）与 /list（只列）共用。"""
    if action in ("/list", "/list-sessions"):
        _print_sessions(ctx.store, console, current=ctx.session_id)
        return
    sessions = ctx.store.list_sessions()
    if not sessions:
        console.print("[yellow]暂无历史会话。[/yellow]")
        return
    _print_sessions(ctx.store, console)
    choice = prompt("选择编号（回车取消）> ").strip()
    if choice.isdigit():
        n = int(choice)
        if 1 <= n <= len(sessions):
            ctx.session_id = sessions[n - 1].id
            _remember_session(last_path, ctx.session_id)
            console.print(f"[dim]已切换到会话 {ctx.session_id[:8]}[/dim]")


def _delete_inline(
    ctx: SessionContext, console: Console, prompt: Callable[[str], str]
) -> None:
    """/delete：按编号删除会话（当前会话不允许删除）。"""
    sessions = ctx.store.list_sessions()
    if not sessions:
        console.print("[yellow]暂无历史会话。[/yellow]")
        return
    _print_sessions(ctx.store, console, current=ctx.session_id)
    choice = prompt("输入要删除的编号（回车取消）> ").strip()
    if not choice.isdigit():
        return
    n = int(choice)
    if not (1 <= n <= len(sessions)):
        console.print("[red]无效编号。[/red]")
        return
    target = sessions[n - 1]
    if target.id == ctx.session_id:
        console.print("[red]当前会话不能删除（/new 后再删）。[/red]")
        return
    ctx.store.delete_session(target.id)
    console.print(f"[dim]已删除会话 {target.id[:8]}[/dim]")


def _plan_inline(ctx: SessionContext, console: Console, redo: bool = False) -> None:
    """/plan：查看当前任务计划；/plan new（redo）清除计划，下次对话重新规划。"""
    if redo:
        ctx.store.clear_plan(ctx.session_id)
        ctx.plan = None
        console.print("[dim]已清除计划；下一条消息将重新规划[/dim]")
        return
    plan = plan_from_messages(ctx.store.get_history(ctx.session_id))
    if plan is None:
        console.print(
            "[yellow]当前会话没有任务计划（简单任务无需计划；多步任务会在首轮生成）。[/yellow]"
        )
        return
    icons = {"pending": "○", "running": "▶", "done": "✓", "failed": "✗"}
    console.print("[bold]任务计划：[/bold]")
    for i, step in enumerate(plan.steps, 1):
        icon = icons.get(step.status, "·")
        console.print(f"  {i}. {icon} {step.description}", markup=False)


def _compact_inline(ctx: SessionContext, console: Console) -> None:
    """/compact：Summarize 压缩当前会话，结果作为后续发送底稿（SQLite 全量历史不动）。"""
    msgs = ctx.store.get_history(ctx.session_id)
    if len(msgs) <= 1:
        console.print("[yellow]会话太短，无需压缩。[/yellow]")
        return
    console.print("[dim]正在压缩历史（LLM 摘要）……[/dim]")
    compacted = ctx.engine.compress(msgs, strategy="summarize", force=True)
    if compacted is msgs:
        console.print("[yellow]没有可压缩的内容（历史已在保护范围内）。[/yellow]")
        return
    ctx.engine.compacted = compacted
    console.print(f"[dim]已压缩：{len(msgs)} 条 → {len(compacted)} 条（对后续对话生效）[/dim]")


if __name__ == "__main__":
    sys.exit(main())
