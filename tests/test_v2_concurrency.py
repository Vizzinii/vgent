"""全方位测试方案 v2 · E 组：并发与竞态。

1. web 同会话并发两个 POST /messages → hub.lock 串行（无交错）
2. web DELETE 正在跑的会话 → 孤儿 messages 行为【已知行为锁定】
3. store WAL 下读并发写 50 次 → 无异常、最终一致
4. episodic 8 线程 × 20 条并发 add + 并发 search → 无丢行、无半行
5. pipeline submit 与 invalidate 并发竞争（epoch 竞态冒烟）
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from conftest import StageClient

from vgent.config import Config
from vgent.llm import ChatResult
from vgent.memory.episodic import EpisodicMemory
from vgent.memory.pipeline import MemoryPipeline
from vgent.memory.store import MemoryFileStore
from vgent.messages import Message, Usage
from vgent.store import SessionStore
from vgent.tools import default_tools
from vgent.web.server import HubManager, make_server


def _start_server(manager: HubManager):
    httpd = make_server(manager, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _post(base: str, path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


class _SerialProbeLLM:
    """慢速 LLM：记录并发进入的峰值（验证 hub.lock 串行）。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        with self.lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.3)
        with self.lock:
            self.active -= 1
        return ChatResult(messages=[Message("assistant", "回复")], usage=Usage(10, 5, 15))


def test_web_same_session_concurrent_posts_serialized(tmp_path) -> None:
    """E1：同会话并发两个 POST /messages → hub.lock 串行（峰值并发=1，消息不交错）。"""
    llm = _SerialProbeLLM()
    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    manager = HubManager(cfg, store, llm, default_tools())
    sid = store.create_session()
    httpd, base = _start_server(manager)
    try:
        results: dict = {}

        def fire(tag: str):
            results[tag] = _post(base, f"/api/sessions/{sid}/messages", {"content": f"消息{tag}"})

        t1 = threading.Thread(target=fire, args=("A",))
        t2 = threading.Thread(target=fire, args=("B",))
        t1.start()
        time.sleep(0.05)  # 让 A 先进入 turn
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        assert not t1.is_alive() and not t2.is_alive()
        assert results["A"][0] == 200 and results["B"][0] == 200
        assert llm.peak == 1, f"同会话并发进入 run_turn：峰值 {llm.peak}"
        # 消息不交错：两个 user 的回复各自成对相邻
        hist = [(m.role, m.content) for m in store.get_history(sid)]
        users = [c for r, c in hist if r == "user"]
        assert sorted(users) == ["消息A", "消息B"]
        for i, (r, c) in enumerate(hist):
            if r == "user":
                assert hist[i + 1] == ("assistant", "回复"), "user 与其回复必须相邻（不交错）"
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_web_delete_running_session_orphans_messages(tmp_path) -> None:
    """E2【已知行为锁定】：DELETE 正在跑的会话 → turn 继续写 store（FK 未启用
    不报错），留下孤儿 messages 行——现状如此（评审记录：不启用外键级联，
    旧库迁移成本），锁定防无意识回归。"""
    llm = _SerialProbeLLM()
    cfg = Config(data_dir=tmp_path)
    store = SessionStore(tmp_path / "t.db")
    manager = HubManager(cfg, store, llm, default_tools())
    sid = store.create_session()
    httpd, base = _start_server(manager)
    try:
        done = threading.Event()

        def fire():
            _post(base, f"/api/sessions/{sid}/messages", {"content": "跑长一点"})
            done.set()

        t = threading.Thread(target=fire, daemon=True)
        t.start()
        time.sleep(0.1)  # turn 进行中
        req = urllib.request.Request(f"{base}/api/sessions/{sid}", method="DELETE")
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 200
        assert done.wait(timeout=30)  # turn 正常完成（不崩）
        assert store.get_title(sid) is None  # sessions 行已删
        # 孤儿 messages 残留【行为锁定】：DELETE 时已落库的 user 被清，turn 后半程
        # 继续写入的 assistant 成为孤儿行（FK 未启用不报错）
        assert len(store.get_history(sid)) >= 1
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_store_concurrent_read_write_50(tmp_path) -> None:
    """E3：WAL 下读（get_history）与写（add_message）并发 50 次 → 无异常、最终一致。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    errors: list[Exception] = []
    stop = threading.Event()

    def writer():
        try:
            for i in range(50):
                store.add_message(sid, Message("user", f"m{i}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            while not stop.is_set():
                store.get_history(sid)
                time.sleep(0.001)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    r = threading.Thread(target=reader)
    w = threading.Thread(target=writer)
    r.start()
    w.start()
    w.join(timeout=30)
    stop.set()
    r.join(timeout=10)
    assert not errors, errors
    hist = store.get_history(sid)
    assert len(hist) == 50 and hist[-1].content == "m49"
    store.close()


def test_episodic_8x20_concurrent_add_and_search(tmp_path) -> None:
    """E4：8 线程 × 20 条并发 add + 并发 search → 无丢行、无半行。"""
    mem = EpisodicMemory(tmp_path / "mem.jsonl")

    def add_worker(tag: int):
        for i in range(20):
            mem.add(f"主题{tag}", f"摘要内容{tag}-{i}" + "详" * 10, f"s-{tag}", f"标题{tag}")

    def search_worker(stop: threading.Event, errors: list):
        try:
            while not stop.is_set():
                mem.search("主题1", limit=5)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    errors: list[Exception] = []
    stop = threading.Event()
    searchers = [threading.Thread(target=search_worker, args=(stop, errors)) for _ in range(2)]
    adders = [threading.Thread(target=add_worker, args=(t,)) for t in range(8)]
    for s in searchers:
        s.start()
    for a in adders:
        a.start()
    for a in adders:
        a.join(timeout=30)
    stop.set()
    for s in searchers:
        s.join(timeout=10)
    assert not errors, errors
    assert mem.count() == 160  # 8×20 无丢行
    # 无半行（每行都能被 from_line 解析 → count 已验证；再抽验结构完整）
    lines = (tmp_path / "mem.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 160
    for line in lines:
        json.loads(line)  # 全部是完整 JSON（无半行）


def test_pipeline_submit_vs_invalidate_race_smoke(tmp_path) -> None:
    """E5：submit 与 invalidate 并发竞争冒烟——epoch 语义下不崩、不写坏文件。"""
    mem_store = MemoryFileStore(tmp_path, tmp_path)
    client = StageClient(stage2_mode="ok")

    class SlowClient(StageClient):
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            time.sleep(0.05)
            return super().chat(messages, tools, on_delta, on_reasoning)

    client = SlowClient()
    pipe = MemoryPipeline(mem_store, client, "m", consolidate_min_signals=5, consolidate_idle_seconds=9999)

    def spam_submit():
        for i in range(10):
            pipe.submit(type("RC", (), {"workspace": "/w", "session_id": "s1", "user_text": f"第{i}轮重要决策内容", "assistant_texts": (), "tool_calls": (), "tool_outputs": ()})())
            time.sleep(0.01)

    t = threading.Thread(target=spam_submit)
    t.start()
    for _ in range(3):  # 并发 invalidate（/memory clear 竞态）
        pipe.invalidate()
        time.sleep(0.02)
    t.join(timeout=10)
    pipe.drain()  # 不崩即过（epoch 竞态冒烟）
    # 文件层未被写坏（可解析或为空模板）
    summary = mem_store.read_summary()
    assert isinstance(summary, str)
