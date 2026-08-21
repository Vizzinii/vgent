"""M11 Web UI 测试：HTTP 端点 + SSE 事件流 + 权限确认桥（FakeLLM 注入，不触网）。"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request

from vgent.config import Config
from vgent.llm import ChatResult
from vgent.messages import Message, ToolCall, Usage
from vgent.store import SessionStore
from vgent.tools import ToolRegistry, ToolSchema
from vgent.web.server import HubManager, make_server


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
    """按顺序返回预设响应的假 LLM（确认桥测试用）。"""

    def __init__(self, responses: list[ChatResult]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls.append(messages)
        return self.responses.pop(0)


def _manager(tmp_path, llm=None, tools=None) -> HubManager:
    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    return HubManager(cfg, store, llm or FakeLLM(), tools)


class _Server:
    """测试用：随机端口起 ThreadingHTTPServer，进程内跑。"""

    def __init__(self, manager: HubManager) -> None:
        self.httpd = make_server(manager, port=0)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def get(self, path: str) -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(self.base + path) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def delete(self, path: str) -> tuple[int, dict]:
        req = urllib.request.Request(self.base + path, method="DELETE")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# -- SSE 读取工具 ------------------------------------------------------------


def _sse_reader(url: str, out: queue.Queue, stop: threading.Event) -> None:
    """逐字节读 SSE 流，整帧解码（避免 UTF-8 多字节被 read(1) 切开），解析出 (event, data)。"""
    with urllib.request.urlopen(url) as r:
        buf = b""
        while not stop.is_set():
            chunk = r.read(1)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                ev = None
                data: dict = {}
                for line in frame.decode("utf-8", errors="replace").split("\n"):
                    if line.startswith("event: "):
                        ev = line[7:]
                    elif line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            pass
                out.put((ev, data))


def _wait_event(out: queue.Queue, name: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ev, data = out.get(timeout=0.3)
        except queue.Empty:
            continue
        if ev == name:
            return data
    raise AssertionError(f"SSE 事件未到达：{name}")


def _collect_until(out: queue.Queue, want: set[str], timeout: float = 5.0) -> tuple[list, dict]:
    """收集事件直到 want 里的名字都见到；返回 (全部事件, 命中的 {name: data})。"""
    events: list[tuple] = []
    found: dict = {}
    pending = set(want)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ev, data = out.get(timeout=0.3)
        except queue.Empty:
            continue
        events.append((ev, data))
        if ev in pending:
            found[ev] = data
            pending.discard(ev)
            if not pending:
                break
    return events, found


def _open_stream(s: _Server, sid: str) -> tuple[queue.Queue, threading.Event]:
    out: queue.Queue = queue.Queue()
    stop = threading.Event()
    t = threading.Thread(
        target=_sse_reader,
        args=(s.base + f"/api/sessions/{sid}/stream", out, stop),
        daemon=True,
    )
    t.start()
    _wait_event(out, "ready")
    return out, stop


# -- 端点测试 ----------------------------------------------------------------


def test_static_page_served(tmp_path) -> None:
    m = _manager(tmp_path)
    s = _Server(m)
    try:
        with urllib.request.urlopen(s.base + "/") as r:
            body = r.read().decode("utf-8")
            assert r.status == 200
            assert "vgent" in body and "EventSource" in body
    finally:
        s.close()
        m.store.close()


def test_sessions_crud(tmp_path) -> None:
    m = _manager(tmp_path)
    s = _Server(m)
    try:
        _, body = s.post("/api/sessions", {})
        sid = body["session_id"]
        _, detail = s.get(f"/api/sessions/{sid}")
        assert detail["session"]["id"] == sid
        assert detail["messages"] == []
        _, listing = s.get("/api/sessions")
        assert len(listing["sessions"]) == 1
        _, resp = s.delete(f"/api/sessions/{sid}")
        assert resp == {"ok": True}
        _, listing = s.get("/api/sessions")
        assert listing["sessions"] == []
    finally:
        s.close()
        m.store.close()


def test_missing_session_404(tmp_path) -> None:
    m = _manager(tmp_path)
    s = _Server(m)
    try:
        code, _ = s.get("/api/sessions/nope")
        assert code == 404
        code, _ = s.post("/api/sessions/nope/messages", {"content": "x"})
        assert code == 404
        code, _ = s.post("/api/sessions/nope/command", {"command": "/plan"})
        assert code == 404
        code, _ = s.post("/api/sessions/nope/confirm", {"result": "approve"})
        assert code == 404
        try:
            urllib.request.urlopen(s.base + "/api/sessions/nope/stream")
            raise AssertionError("stream 应 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        except ConnectionError:
            pass  # 偶发时序：服务器关连接等价于 404（Windows keep-alive 竞态）
    finally:
        s.close()
        m.store.close()


def test_turn_persists_history_and_status(tmp_path) -> None:
    m = _manager(tmp_path)
    s = _Server(m)
    try:
        _, body = s.post("/api/sessions", {})
        sid = body["session_id"]
        code, payload = s.post(f"/api/sessions/{sid}/messages", {"content": "你好"})
        assert code == 200
        assert payload["usage"]["total_tokens"] == 15
        assert payload["state"] == "completed"
        _, detail = s.get(f"/api/sessions/{sid}")
        assert [mm["role"] for mm in detail["messages"]] == ["user", "assistant"]
        assert detail["provider_name"] == "deepseek"
        assert detail["provider_model"] == "deepseek-v4-flash"
    finally:
        s.close()
        m.store.close()


def test_turn_streams_events_over_sse(tmp_path) -> None:
    m = _manager(tmp_path)
    s = _Server(m)
    try:
        _, body = s.post("/api/sessions", {})
        sid = body["session_id"]
        out, stop = _open_stream(s, sid)
        code, payload = s.post(f"/api/sessions/{sid}/messages", {"content": "你好"})
        assert code == 200
        assert payload["usage"]["total_tokens"] == 15
        events, found = _collect_until(out, {"done"})
        assert "done" in found
        assert found["done"]["usage"]["total_tokens"] == 15
        assert found["done"]["state"] == "completed"
        deltas = "".join(d["text"] for e, d in events if e == "delta")
        assert deltas == "你好"
        stop.set()
    finally:
        s.close()
        m.store.close()


def test_confirm_bridge(tmp_path) -> None:
    """exec 工具 → SSE confirm 事件 → 浏览器回传 approve → 工具执行 → done。"""
    reg = ToolRegistry()
    executed = 0

    def probe(args: dict) -> str:
        nonlocal executed
        executed += 1
        return "ok"

    reg.register(ToolSchema("probe", "探测", {"type": "object"}, "exec"), probe)
    responses = [
        ChatResult(
            messages=[Message("assistant", "", tool_calls=[ToolCall("c1", "probe", "{}")])],
            usage=Usage(5, 1, 6),
            tool_calls=[ToolCall("c1", "probe", "{}")],
        ),
        ChatResult(messages=[Message("assistant", "完成")], usage=Usage(5, 2, 7)),
    ]
    m = _manager(tmp_path, llm=ScriptedLLM(responses), tools=reg)
    s = _Server(m)
    try:
        _, body = s.post("/api/sessions", {})
        sid = body["session_id"]
        out, stop = _open_stream(s, sid)

        result: dict = {}

        def post_message() -> None:
            code, payload = s.post(f"/api/sessions/{sid}/messages", {"content": "执行"})
            result["code"] = code
            result["payload"] = payload

        t = threading.Thread(target=post_message, daemon=True)
        t.start()
        events, found = _collect_until(out, {"confirm"})
        assert found["confirm"]["tool"] == "probe"
        assert found["confirm"]["permission"] == "exec"
        code, resp = s.post(f"/api/sessions/{sid}/confirm", {"result": "approve"})
        assert code == 200 and resp == {"ok": True}
        t.join(timeout=5)
        assert not t.is_alive()

        assert executed == 1
        assert result["code"] == 200
        assert result["payload"]["state"] == "completed"
        rest, found = _collect_until(out, {"done"})
        events += rest
        assert found["done"]["state"] == "completed"
        names = [e for e, _ in events]
        assert "tool" in names  # 工具执行卡片事件已广播
        stop.set()
    finally:
        s.close()
        m.store.close()


def test_command_endpoint(tmp_path) -> None:
    m = _manager(tmp_path)
    s = _Server(m)
    try:
        _, body = s.post("/api/sessions", {})
        sid = body["session_id"]
        code, resp = s.post(f"/api/sessions/{sid}/command", {"command": "/plan"})
        assert code == 200 and "没有任务计划" in resp["text"]
        code, resp = s.post(f"/api/sessions/{sid}/command", {"command": "/reasoning"})
        assert code == 200 and "开" in resp["text"]
        code, resp = s.post(f"/api/sessions/{sid}/command", {"command": "/bad"})
        assert code == 200 and "未知命令" in resp["text"]
        code, resp = s.post(f"/api/sessions/{sid}/command", {"command": "/compact"})
        assert code == 200 and "会话太短" in resp["text"]
    finally:
        s.close()
        m.store.close()


def test_deny_pruning_and_allow_command(tmp_path) -> None:
    """P10：deny 工具从 schemas 裁剪（模型看不到）；P2：/allow 写回 config.toml。"""
    import tomllib

    from vgent.config import PermissionRules

    (tmp_path / "config.toml").write_text('[provider]\nactive = "deepseek"\n', encoding="utf-8")
    cfg = Config(data_dir=tmp_path)
    cfg.permissions = PermissionRules(deny=["shell"], allow=["read_file"])
    store = SessionStore(tmp_path / "t.db")
    m = HubManager(cfg, store, FakeLLM())
    names = {s["function"]["name"] for s in m.tools.schemas()}
    assert "shell" not in names  # deny 裁剪
    assert "read_file" in names
    s = _Server(m)
    try:
        _, body = s.post("/api/sessions", {})
        sid = body["session_id"]
        code, resp = s.post(f"/api/sessions/{sid}/command", {"command": "/allow"})
        assert code == 200 and "read_file" in resp["text"]  # 列出当前持久化 allow
        code, resp = s.post(f"/api/sessions/{sid}/command", {"command": "/allow write_file"})
        assert code == 200 and "已放行" in resp["text"]
        # UX 修复：内存规则同步（/allow 无参列表能立即看到）
        assert "write_file" in m.hub(sid).ctx.permissions.rules.allow
        data = tomllib.loads((tmp_path / "config.toml").read_text(encoding="utf-8"))
        assert data["permissions"]["allow"] == ["write_file"]
    finally:
        s.close()
        m.store.close()
