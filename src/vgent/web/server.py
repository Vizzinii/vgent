"""M11 Web UI —— 本地 HTTP 服务（stdlib 零依赖）+ SSE 桥接 sync 引擎。

形态：`vgent --serve`（或 `vgent serve`）起 127.0.0.1 上的 ThreadingHTTPServer，
浏览器打开单文件前端（web/static/index.html）。CLI 保留双入口。

桥接思路（不动 agent.py 的 sync 引擎）：
- 每会话一个 SessionHub：持有 SessionContext + 每会话锁（同一会话串行）；
- POST /messages → 在请求线程里直接跑 run_turn（阻塞，流式事件经 SSE 广播）；
  浏览器先开 EventSource 订阅该会话的事件流；
- 权限确认（契约③）：PermissionSystem.confirm 注入「SSE 事件 + 阻塞队列」，
  浏览器模态框 POST /confirm 回传 y/a/n（含 sticky，复用三档语义）；
- 命令（/plan /compact /reasoning /remember 等）：POST /command，逻辑与
  cli.py 的 _dispatch_command 同语义，只返回文本（前端展示）。

已知边界（M11 立项时声明）：单用户 localhost、无需鉴权；同一会话串行；
rich 渲染在浏览器侧由原生样式替代。
"""
from __future__ import annotations

import json
import os
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from vgent.agent import SessionContext, run_turn
from vgent.config import Config
from vgent.context import ContextEngine
from vgent.llm import ChatResult, LLMClient
from vgent.mcp import load_into_registry
from vgent.memory.episodic import EpisodicMemory, summarize
from vgent.messages import Message
from vgent.permission import ConfirmResult, PermissionSystem
from vgent.store import SessionStore
from vgent.task import plan_from_messages
from vgent.tools import ToolRegistry, default_tools
from vgent.workspace import find_instructions

DEFAULT_PORT = 8477
CONFIRM_TIMEOUT = 600.0  # 确认弹窗最长等待（秒）；超时按拒绝处理（防挂死 turn）
HEARTBEAT = 15.0  # SSE 心跳间隔（秒），保持连接不被中间层掐断
_TOOL_SUMMARY_CAP = 120  # 工具结果状态行首行截断（与 CLI 状态行口径一致）

COMMAND_HELP = """命令：
  /plan            查看任务计划（/plan new 清除并重新规划）
  /compact         压缩当前会话（LLM 摘要中间历史，下次对话生效）
  /reasoning       切换思考过程展示（开/关）
  /remember <主题> 记住当前会话（LLM 摘要存本机）
  /recall <关键词> 检索历史记忆并注入上下文
  /memories        列出已记住的任务摘要
  /mcp             列出已加载的 MCP 工具
  /help            显示帮助
"""


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class SessionHub:
    """一个浏览器会话的服务端状态：SessionContext + SSE 订阅 + 确认队列。"""

    def __init__(self, ctx: SessionContext) -> None:
        self.ctx = ctx
        self.lock = threading.RLock()  # 同一会话串行跑 turn（M11 边界声明）
        self.running = False
        self._clients: list[queue.Queue] = []
        self._clients_lock = threading.Lock()
        self._confirm_q: queue.Queue | None = None

    # -- SSE 订阅 ----------------------------------------------------------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._clients_lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._clients_lock:
            if q in self._clients:
                self._clients.remove(q)

    def broadcast(self, event: str, data: dict) -> None:
        frame = _sse(event, data)
        with self._clients_lock:
            clients = list(self._clients)
        for q in clients:
            q.put(frame)

    # -- 权限确认桥（契约③，PermissionSystem.confirm 注入点）-----------------

    def confirm(self, tool, args) -> ConfirmResult:
        """SSE 发确认事件 → 阻塞等浏览器回传；超时/无浏览器按拒绝处理。"""
        self._confirm_q = queue.Queue()
        self.broadcast("confirm", {"tool": tool.name, "permission": tool.permission, "args": args})
        try:
            ans = self._confirm_q.get(timeout=CONFIRM_TIMEOUT)
        except queue.Empty:
            return ConfirmResult.REJECT
        finally:
            self._confirm_q = None
        return {
            "approve": ConfirmResult.APPROVE,
            "always": ConfirmResult.ALWAYS,
            "reject": ConfirmResult.REJECT,
        }.get(ans, ConfirmResult.REJECT)

    def resolve_confirm(self, result: str) -> bool:
        """POST /confirm：把浏览器选择投递给阻塞中的 turn 线程。"""
        if self._confirm_q is None:
            return False
        self._confirm_q.put(result)
        return True

    # -- 跑一轮 -------------------------------------------------------------

    def run_turn_web(self, text: str) -> dict:
        """在调用线程里跑 run_turn（POST /messages 的请求线程），事件走 SSE。"""
        with self.lock:
            self.running = True
            self.broadcast("status", {"running": True})
            try:
                result = run_turn(
                    text,
                    self.ctx,
                    on_delta=lambda d: self.broadcast("delta", {"text": d}),
                    on_reasoning=(
                        (lambda d: self.broadcast("reasoning", {"text": d}))
                        if self.ctx.show_reasoning
                        else None
                    ),
                    on_tool=lambda n, o: self.broadcast(
                        "tool", {"name": n, "summary": _tool_summary(o)}
                    ),
                )
                payload = self._result_payload(result)
                self.broadcast("done", payload)
                return payload
            except Exception as exc:  # noqa: BLE001 — 失败也要让页面知道，不崩服务
                self.broadcast("error", {"message": str(exc)})
                return {"error": str(exc)}
            finally:
                self.running = False
                self.broadcast("status", {"running": False})

    def _result_payload(self, result: ChatResult) -> dict:
        usage = None
        if result.usage:
            usage = {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            }
        return {
            "usage": usage,
            "state": self.ctx.state.value,
            "plan": _plan_json(self.ctx.plan),
            "compression_count": self.ctx.engine.compression_count,
            "message_count": len(self.ctx.store.get_history(self.ctx.session_id)),
        }


class HubManager:
    """启动时接线一次（store/llm/tools/mcp/memory/instructions，与 cli.main 同源），
    按会话懒建 SessionHub（每会话独立 engine/permissions/sticky）。"""

    def __init__(
        self,
        cfg: Config,
        store: SessionStore,
        llm: LLMClient | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.llm = llm or LLMClient(cfg)
        self.tools = tools or default_tools()
        self.mcp_loaded = load_into_registry(self.tools, cfg.mcp_servers)
        self.memory = EpisodicMemory(cfg.data_dir / "memory" / "episodic.jsonl")
        found = find_instructions(os.getcwd())
        self.instructions = found[1] if found else None
        self.instructions_name = found[0] if found else None
        self._hubs: dict[str, SessionHub] = {}
        self._lock = threading.Lock()

    def _summarizer(self, middle: list[Message]) -> str:
        """Summarize 策略的 LLM 摘要器（/compact 用，与 cli 同口径）。"""
        prompt = Message(
            "system",
            "你是会话压缩器。把下面的对话历史压缩成 3~5 句要点摘要，"
            "保留关键事实、已做的决定和未完成的任务；只输出摘要本身。",
        )
        result = self.llm.chat([prompt, *middle])
        return (result.messages[0].content or "").strip()

    def hub(self, sid: str) -> SessionHub:
        with self._lock:
            hub = self._hubs.get(sid)
            if hub is None:
                engine = ContextEngine(self.cfg.provider.context_length, self.cfg.context)
                engine.summarizer = self._summarizer
                ctx = SessionContext(
                    session_id=sid,
                    store=self.store,
                    llm=self.llm,
                    tools=self.tools,
                    engine=engine,
                    show_reasoning=self.cfg.show_reasoning,
                    memory=self.memory,
                    memory_auto=self.cfg.memory_auto,
                    mcp_tools=self.mcp_loaded,
                    instructions=self.instructions,
                    instructions_name=self.instructions_name,
                )
                hub = SessionHub(ctx)
                ctx.permissions = PermissionSystem(confirm=hub.confirm)
                self._hubs[sid] = hub
            return hub

    def drop(self, sid: str) -> None:
        with self._lock:
            self._hubs.pop(sid, None)


# -- 命令（与 cli.py _dispatch_command 同语义，返回文本供前端展示）---------------


def run_command(text: str, hub: SessionHub) -> str:
    ctx = hub.ctx
    text = text.strip()
    if text == "/help":
        return COMMAND_HELP
    if text == "/reasoning":
        ctx.show_reasoning = not ctx.show_reasoning
        return f"思考过程展示：{'开' if ctx.show_reasoning else '关'}"
    if text == "/compact":
        return _cmd_compact(hub)
    if text == "/plan" or text.startswith("/plan "):
        return _cmd_plan(hub, redo=("new" in text or "redo" in text))
    if text == "/memories" or text == "/remember":
        return _cmd_memories(hub)
    if text.startswith("/remember "):
        return _cmd_remember(hub, text[len("/remember ") :].strip())
    if text.startswith("/recall "):
        return _cmd_recall(hub, text[len("/recall ") :].strip())
    if text == "/mcp":
        return _cmd_mcp(hub)
    return f"未知命令：{text}（/help 查看）"


def _cmd_compact(hub: SessionHub) -> str:
    ctx = hub.ctx
    msgs = ctx.store.get_history(ctx.session_id)
    if len(msgs) <= 1:
        return "会话太短，无需压缩。"
    compacted = ctx.engine.compress(msgs, strategy="summarize", force=True)
    if compacted is msgs:
        return "没有可压缩的内容（历史已在保护范围内）。"
    ctx.engine.compacted = compacted
    return f"已压缩：{len(msgs)} 条 → {len(compacted)} 条（对后续对话生效）"


def _cmd_plan(hub: SessionHub, redo: bool) -> str:
    ctx = hub.ctx
    if redo:
        ctx.store.clear_plan(ctx.session_id)
        ctx.plan = None
        return "已清除计划；下一条消息将重新规划"
    plan = plan_from_messages(ctx.store.get_history(ctx.session_id))
    if plan is None:
        return "当前会话没有任务计划（简单任务无需计划；多步任务会在首轮生成）。"
    icons = {"pending": "○", "running": "▶", "done": "✓", "failed": "✗"}
    lines = ["任务计划："]
    for i, step in enumerate(plan.steps, 1):
        lines.append(f"  {i}. {icons.get(step.status, '·')} {step.description}")
    return "\n".join(lines)


def _cmd_remember(hub: SessionHub, topic: str) -> str:
    ctx = hub.ctx
    if ctx.memory is None:
        return "记忆未启用。"
    if not topic:
        return "用法：/remember <主题>"
    msgs = ctx.store.get_history(ctx.session_id)
    if len(msgs) < 2:
        return "会话太短，无可整理内容。"
    summary = summarize(msgs, ctx.llm, topic)
    if not summary:
        return "记忆整理失败（LLM 无响应）。"
    title = ctx.store.get_title(ctx.session_id) or topic
    ctx.memory.add(topic, summary, ctx.session_id, title)
    return f"已记住（{topic}）：{summary.splitlines()[0][:80]}"


def _cmd_recall(hub: SessionHub, keyword: str) -> str:
    ctx = hub.ctx
    if ctx.memory is None:
        return "记忆未启用。"
    if not keyword:
        return "用法：/recall <关键词>"
    hits = ctx.memory.search(keyword, limit=3)
    if not hits:
        return f"没有匹配「{keyword}」的历史记忆。"
    for e in hits:
        ctx.store.add_message(
            ctx.session_id,
            Message("system", f"[记忆] {e.topic}（{e.ts[:10]}）：{e.summary}"),
        )
    return f"已注入 {len(hits)} 条记忆（后续对话可见）。"


def _cmd_memories(hub: SessionHub) -> str:
    ctx = hub.ctx
    if ctx.memory is None:
        return "记忆未启用。"
    entries = ctx.memory.list_recent(10)
    if not entries:
        return "还没有任何历史记忆（用 /remember <主题> 记住当前会话）。"
    lines = ["历史记忆："]
    for e in entries:
        first = e.summary.splitlines()[0] if e.summary else ""
        lines.append(f"  {e.ts[:16]} [{e.topic}] {first[:60]}")
    return "\n".join(lines)


def _cmd_mcp(hub: SessionHub) -> str:
    ctx = hub.ctx
    if not ctx.mcp_tools:
        return "未配置 MCP 服务器（config.toml 的 [mcp.servers.<name>]，command/args 指向本地 server）。"
    lines = ["已加载的 MCP 工具："]
    for server, names in ctx.mcp_tools.items():
        if names:
            lines.append(f"  {server}: {', '.join(names)}")
        else:
            lines.append(f"  {server}: 加载失败（启动时已跳过）")
    return "\n".join(lines)


# -- HTTP 层（stdlib，零依赖）------------------------------------------------


class WebHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    manager: HubManager  # 由 make_server 挂载

    # -- 工具方法 -----------------------------------------------------------

    def _send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _send_static(self) -> None:
        path = Path(__file__).parent / "static" / "index.html"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _parts(path: str) -> list[str]:
        return [unquote(p) for p in urlparse(path).path.split("/") if p]

    def log_message(self, fmt: str, *args) -> None:  # 静默请求日志，避免刷屏
        pass

    # -- 路由 ---------------------------------------------------------------

    def do_GET(self) -> None:
        try:
            parts = self._parts(self.path)
            if not parts or parts[0] != "api":
                return self._send_static()
            self._api_get(parts)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001 — 任何处理失败都回 JSON，不崩连接
            self._send_json(500, {"error": f"server error: {exc}"})

    def do_POST(self) -> None:
        try:
            parts = self._parts(self.path)
            if not parts or parts[0] != "api":
                return self._send_json(404, {"error": "not found"})
            self._api_post(parts)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"server error: {exc}"})

    def do_DELETE(self) -> None:
        try:
            parts = self._parts(self.path)
            if parts[:2] == ["api", "sessions"] and len(parts) == 3:
                m = self.manager
                m.drop(parts[2])
                m.store.delete_session(parts[2])
                return self._send_json(200, {"ok": True})
            self._send_json(404, {"error": "not found"})
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": f"server error: {exc}"})

    # -- API ---------------------------------------------------------------

    def _api_get(self, parts: list[str]) -> None:
        m = self.manager
        if parts[1:] == ["sessions"]:
            sessions = [
                {
                    "id": s.id,
                    "title": s.title,
                    "created_at": s.created_at,
                    "message_count": s.message_count,
                    "state": m.store.get_state(s.id),
                }
                for s in m.store.list_sessions()
            ]
            return self._send_json(200, {"sessions": sessions})
        if parts[1] == "sessions" and len(parts) >= 3:
            sid = parts[2]
            if len(parts) == 4 and parts[3] == "stream":
                return self._stream(sid)
            if m.store.get_title(sid) is None:
                return self._send_json(404, {"error": "session not found"})
            hub = m.hub(sid)
            return self._send_json(
                200,
                {
                    "session": {"id": sid, "title": m.store.get_title(sid)},
                    "state": m.store.get_state(sid),
                    "messages": [_msg_json(x) for x in m.store.get_history(sid)],
                    "plan": _plan_json(hub.ctx.plan),
                    "running": hub.running,
                    "provider_name": m.cfg.provider.name,
                    "provider_model": m.cfg.provider.model,
                },
            )
        self._send_json(404, {"error": "not found"})

    def _api_post(self, parts: list[str]) -> None:
        m = self.manager
        if parts[1:] == ["sessions"]:
            return self._send_json(200, {"session_id": m.store.create_session()})
        if parts[1] == "sessions" and len(parts) == 4 and parts[3] in (
            "messages",
            "confirm",
            "command",
        ):
            sid = parts[2]
            if m.store.get_title(sid) is None:
                return self._send_json(404, {"error": "session not found"})
            hub = m.hub(sid)
            body = self._read_json()
            if parts[3] == "messages":
                text = str(body.get("content", "")).strip()
                if not text:
                    return self._send_json(400, {"error": "empty content"})
                payload = hub.run_turn_web(text)  # 阻塞到本轮结束（SSE 已推流）
                return self._send_json(200 if "error" not in payload else 500, payload)
            if parts[3] == "confirm":
                result = str(body.get("result", "reject"))
                if result not in ("approve", "always", "reject"):
                    return self._send_json(400, {"error": "bad result"})
                return self._send_json(200, {"ok": hub.resolve_confirm(result)})
            if parts[3] == "command":
                cmd = str(body.get("command", "")).strip()
                return self._send_json(200, {"text": run_command(cmd, hub)})
        self._send_json(404, {"error": "not found"})

    def _stream(self, sid: str) -> None:
        """SSE 事件流：连接即发 ready，随后转发该会话 hub 的广播。"""
        m = self.manager
        if m.store.get_title(sid) is None:
            self.close_connection = True  # 404 后关连接：避免 keep-alive 关闭竞态
            return self._send_json(404, {"error": "session not found"})
        hub = m.hub(sid)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        q = hub.subscribe()
        try:
            self.wfile.write(_sse("ready", {"session_id": sid}).encode("utf-8"))
            self.wfile.flush()
            while True:
                try:
                    frame = q.get(timeout=HEARTBEAT)
                    self.wfile.write(frame.encode("utf-8"))
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            hub.unsubscribe(q)


def _tool_summary(out: str) -> str:
    first = out.splitlines()[0] if out else ""
    if len(first) > _TOOL_SUMMARY_CAP:
        first = first[:_TOOL_SUMMARY_CAP] + "…"
    return first


def _msg_json(m: Message) -> dict:
    return {
        "role": m.role,
        "content": m.content,
        "reasoning_content": m.reasoning_content,
        "tool_calls": (
            [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls]
            if m.tool_calls
            else None
        ),
        "tool_call_id": m.tool_call_id,
    }


def _plan_json(plan) -> dict | None:
    if plan is None:
        return None
    return {"steps": [{"description": s.description, "status": s.status} for s in plan.steps]}


def make_server(manager: HubManager, port: int = 0, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """建 ThreadingHTTPServer（port=0 = 随机端口，测试用）。"""
    httpd = ThreadingHTTPServer((host, port), WebHandler)
    httpd.daemon_threads = True
    WebHandler.manager = manager
    return httpd


def serve(cfg: Config, port: int = DEFAULT_PORT, open_browser: bool = True) -> int:
    """`vgent --serve` 入口：接线 + 起服务 + 自动开浏览器。"""
    store = SessionStore(cfg.data_dir / "sessions" / "vgent.db")
    manager = HubManager(cfg, store)
    httpd = make_server(manager, port=port)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    print(f"vgent Web UI：{url}（Ctrl+C 停止）")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001, S110 — 无浏览器环境不阻塞启动
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
        store.close()
    return 0
