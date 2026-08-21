"""M3 测试：ContextEngine（usage 计数 + 低水位剪枝 + TailWindow 压缩 + 孤儿清理）。"""
from __future__ import annotations

from vgent.config import ContextConfig
from vgent.context import ContextEngine
from vgent.messages import Message, ToolCall, Usage


def _tool_msg(content: str, call_id: str = "c1") -> Message:
    return Message("tool", content, tool_call_id=call_id)


def test_default_engine_is_noop() -> None:
    engine = ContextEngine()
    assert not engine.should_compress()
    msgs = [Message("user", "hi")]
    out, n = engine.prune_tool_results_only(msgs)
    assert out is msgs and n == 0
    assert engine.compress(msgs) is msgs


def test_usage_drives_should_compress() -> None:
    engine = ContextEngine(context_length=1000, cfg=ContextConfig())  # 高水位 750
    engine.update_from_response(Usage(400, 100, 500))
    assert not engine.should_compress()
    engine.update_from_response(Usage(800, 100, 900))
    assert engine.should_compress()


def test_prune_noop_below_watermark() -> None:
    engine = ContextEngine(context_length=1000, cfg=ContextConfig(prune_percent=0.5))
    msgs = [Message("user", "hi")]
    out, n = engine.prune_tool_results_only(msgs)
    assert out is msgs and n == 0


def test_prune_summarizes_long_tool_results() -> None:
    engine = ContextEngine(context_length=1000, cfg=ContextConfig(prune_percent=0.01))
    msgs = [
        Message("user", "u0"),
        Message("assistant", "", tool_calls=[ToolCall("c1", "shell", '{"command":"ls"}')]),
        _tool_msg("line1\n" + "y" * 600, "c1"),  # 在受保护尾部之外，会被摘要
        Message("user", "u1"),
        Message("assistant", "a1"),
        Message("user", "u2"),
        Message("assistant", "a2"),
        Message("user", "u3"),
        Message("assistant", "a3"),
    ]
    out, n = engine.prune_tool_results_only(msgs)
    assert n == 1
    assert out[2].role == "tool"
    assert out[2].tool_call_id == "c1"  # 摘要保留配对
    assert "\n" not in out[2].content
    assert "已摘要" in out[2].content
    assert out[2].content == "line1（原 606 字符，已摘要）"
    # 受保护尾部（最近 6 条）未被改动
    assert out[3:] == msgs[3:]
    assert out[1].tool_calls == msgs[1].tool_calls


def test_prune_cleans_orphan_tool_pairs() -> None:
    engine = ContextEngine(context_length=1000, cfg=ContextConfig(prune_percent=0.01))
    msgs = [
        Message("user", "u0"),
        _tool_msg("孤儿" + "z" * 600, "c9"),  # 无 assistant 配对 → 丢弃
        Message("user", "u1"),
        Message("assistant", "", tool_calls=[ToolCall("c2", "shell", "{}")]),  # 无结果 → 丢弃
        Message("user", "u2"),
        Message("assistant", "a2"),
        Message("user", "u3"),
        Message("assistant", "a3"),
        Message("user", "u4"),
    ]
    out, n = engine.prune_tool_results_only(msgs)
    assert n == 1  # 长 tool 结果被摘要（随后在清理中被丢弃）
    assert all(m.tool_call_id != "c9" for m in out)
    assert not any(m.role == "assistant" and m.tool_calls for m in out)


def test_compress_noop_below_threshold() -> None:
    engine = ContextEngine(context_length=1000, cfg=ContextConfig(threshold_percent=0.9))
    msgs = [Message("user", "hi")]
    assert engine.compress(msgs) is msgs


def test_compress_keeps_head_and_tail_with_marker() -> None:
    engine = ContextEngine(
        context_length=1000,
        cfg=ContextConfig(threshold_percent=0.1, tail_token_budget=60),  # 高水位 100
    )
    msgs = [Message("user", "x" * 40) for _ in range(20)]  # 每条估算 17，共 340
    out = engine.compress(msgs)
    assert out[0] is msgs[0]  # 头部保留
    assert out[-1] is msgs[-1]  # 最新保留
    assert out[1].role == "system" and "TailWindow" in out[1].content
    assert "15" in out[1].content  # 中间 15 条被压缩
    assert len(out) == 6
    assert engine.compression_count == 1


def test_compress_does_not_split_tool_pairs() -> None:
    """尾部预算边界落在 tool 结果上时，连同其 assistant 一起保留。"""
    engine = ContextEngine(
        context_length=1000,
        cfg=ContextConfig(threshold_percent=0.03, tail_token_budget=12),  # 高水位 30
    )
    msgs = [
        Message("user", "u0"),
        Message("user", "u1"),
        Message("user", "u2"),
        Message("assistant", "", tool_calls=[ToolCall("c1", "shell", '{"command":"x"}')]),
        _tool_msg("结果", "c1"),
        Message("user", "u3"),
        Message("assistant", "a3"),
    ]
    out = engine.compress(msgs)
    # assistant c1 与其 tool 结果成对保留
    assert any(m.role == "assistant" and m.tool_calls and m.tool_calls[0].id == "c1" for m in out)
    assert any(m.role == "tool" and m.tool_call_id == "c1" for m in out)
    # 中间段被丢：u1、u2 不在发送列表里
    assert not any(m.content in ("u1", "u2") for m in out)
    # 没有孤儿 tool 结果
    assert all(m.tool_call_id != "c1" or m.role == "tool" for m in out)


def test_compress_hard_floor_keeps_last() -> None:
    engine = ContextEngine(
        context_length=30,
        cfg=ContextConfig(threshold_percent=0.5, tail_token_budget=5),
    )
    msgs = [Message("user", "x" * 40) for _ in range(20)]
    out = engine.compress(msgs)
    assert out[-1] is msgs[-1]  # 硬下限兜底：最后一条永远保留
    assert engine._estimate_tokens(out) <= 30


# -- M4：Summarize 策略 + /compact -------------------------------------------


def test_compress_summarize_uses_summarizer() -> None:
    """cfg.compact_strategy="summarize" + 注入 summarizer：中间段被 LLM 摘要替代。"""
    engine = ContextEngine(
        context_length=1000,
        cfg=ContextConfig(threshold_percent=0.1, tail_token_budget=60, compact_strategy="summarize"),
        summarizer=lambda middle: "要点摘要内容",
    )
    msgs = [Message("user", "x" * 40) for _ in range(20)]
    out = engine.compress(msgs)
    assert out[0] is msgs[0]
    assert out[1].role == "system"
    assert "【历史摘要" in out[1].content
    assert "要点摘要内容" in out[1].content
    assert len(out) == 6


def test_compress_summarize_falls_back_without_summarizer() -> None:
    """没有注入 summarizer：summarize 退回 TailWindow 标记，不报错。"""
    engine = ContextEngine(
        context_length=1000,
        cfg=ContextConfig(threshold_percent=0.1, tail_token_budget=60),
    )
    msgs = [Message("user", "x" * 40) for _ in range(20)]
    out = engine.compress(msgs, strategy="summarize")
    assert "TailWindow" in out[1].content


def test_compress_summarize_falls_back_when_summarizer_raises() -> None:
    """摘要器抛异常（网络等）：退回 TailWindow 标记，压缩不中断。"""

    def boom(middle) -> str:
        raise RuntimeError("api down")

    engine = ContextEngine(
        context_length=1000,
        cfg=ContextConfig(threshold_percent=0.1, tail_token_budget=60),
        summarizer=boom,
    )
    msgs = [Message("user", "x" * 40) for _ in range(20)]
    out = engine.compress(msgs, strategy="summarize")
    assert "TailWindow" in out[1].content


def test_compress_force_bypasses_threshold() -> None:
    """force=True（/compact）：低于水位也压缩。"""
    engine = ContextEngine(
        context_length=1_000_000,
        cfg=ContextConfig(tail_token_budget=60),  # 高水位 750K，历史远低于
    )
    msgs = [Message("user", "x" * 40) for _ in range(20)]
    out = engine.compress(msgs, strategy="tail", force=True)
    assert len(out) < len(msgs)
    assert out[0] is msgs[0] and out[-1] is msgs[-1]


# -- P4：extract_summary ------------------------------------------------------


def test_extract_summary_blocks() -> None:
    from vgent.context import extract_summary

    assert extract_summary("<analysis>草稿</analysis>\n<summary>要点</summary>") == "要点"
    assert extract_summary("<analysis>只有草稿</analysis>") == "<analysis>只有草稿</analysis>"
    assert extract_summary("无块原文") == "无块原文"
    assert extract_summary("") == ""
