"""全方位测试方案 v2 · G 组：规模与性能（阈值断言，防退化）。

1. 5000 条历史 get_history < 0.5s（F16 索引效果）
2. 有 compact 记录的会话连跑 20 轮 < 5s（F1 引入的每轮重建，防性能滑坡）
3. 5MB+1 文件 read_file 立即返回错误（< 0.1s，不整读）
4. search 大目录（1000 小文件 + 10 个跳过目录）< 3s

阈值在慢机器上可能偏紧——失败时先看量级（数量级超标才算回归）。
"""
from __future__ import annotations

import time

from conftest import FakeLLM

from vgent.agent import SessionContext, run_turn
from vgent.messages import Message
from vgent.store import SessionStore
from vgent.tools import _MAX_FILE_BYTES, default_tools


def test_get_history_5000_under_half_second(tmp_path) -> None:
    """G1：5000 条历史 get_history < 0.5s（idx_messages_session 生效）。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    rows = [(sid, "user", f"消息内容第{i}行", None, None, None, f"2026-01-01T00:00:{i % 60:02d}") for i in range(5000)]
    with store._lock:
        store._conn.executemany(
            "INSERT INTO messages (session_id, role, content, reasoning_content, tool_calls, tool_call_id, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        store._conn.commit()
    t0 = time.perf_counter()
    hist = store.get_history(sid)
    elapsed = time.perf_counter() - t0
    assert len(hist) == 5000
    assert elapsed < 0.5, f"get_history 5000 条耗时 {elapsed:.3f}s（索引可能未生效）"
    store.close()


def test_compacted_session_20_turns_under_5s(tmp_path) -> None:
    """G2：有 compact 记录的会话连跑 20 轮 < 5s（每轮 _compacted_from_store 重建）。"""
    store = SessionStore(tmp_path / "t.db")
    sid = store.create_session()
    store.upsert_compact(sid, "历史摘要：" + "要" * 200, [Message("assistant", "尾" * 50)], 1)
    llm = FakeLLM()
    ctx = SessionContext(session_id=sid, store=store, llm=llm)
    t0 = time.perf_counter()
    for i in range(20):
        run_turn(f"第{i}轮", ctx)
    elapsed = time.perf_counter() - t0
    assert len(llm.calls) == 20
    assert elapsed < 5.0, f"20 轮压缩重建耗时 {elapsed:.2f}s（性能滑坡）"
    store.close()


def test_read_oversized_returns_fast(tmp_path) -> None:
    """G3：5MB+1 文件 read_file 立即报错（< 0.1s，不整读进内存）。"""
    big = tmp_path / "big.txt"
    chunk = "x" * 8192
    big.write_text(chunk * 640, encoding="utf-8")  # 5MB
    with big.open("a", encoding="utf-8") as f:
        f.write("x")  # +1 超限
    assert big.stat().st_size > _MAX_FILE_BYTES
    reg = default_tools()
    t0 = time.perf_counter()
    out = reg.execute("read_file", {"path": str(big)})
    elapsed = time.perf_counter() - t0
    assert out.startswith("错误：") and "5MB" in out
    assert elapsed < 0.1, f"超限检测耗时 {elapsed:.3f}s（疑似整读了）"


def test_search_1000_files_under_3s(tmp_path) -> None:
    """G4：search 大目录（1000 小文件 + 10 个跳过目录）< 3s。"""
    for i in range(1000):
        (tmp_path / f"f{i:04}.txt").write_text(f"普通内容 {i}\n", encoding="utf-8")
    for d in (".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache",
              ".ruff_cache", "dist", "build", ".idea"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "noise.txt").write_text("needle noise 不应被搜到\n", encoding="utf-8")
    (tmp_path / "f0500.txt").write_text("needle 目标行\n", encoding="utf-8")  # 造一个命中
    reg = default_tools()
    t0 = time.perf_counter()
    out = reg.execute("search", {"pattern": "needle", "path": str(tmp_path)})
    elapsed = time.perf_counter() - t0
    assert "f0500.txt" in out
    assert "noise.txt" not in out  # 跳过目录不进结果
    assert elapsed < 3.0, f"search 1000 文件耗时 {elapsed:.2f}s"
