"""llm.py 流式累积测试（内容 / reasoning_content / tool_calls 分片 / usage / 重试）。

用假 chunk 注入，不触网。tool_calls 按「index 槽 + id 校验」合并（M5：
DeepSeek 实测分片索引连续；同 index 出现新 id 视为新工具调用，防御交错/复用）。
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from vgent.config import Config
from vgent.llm import LLMClient
from vgent.messages import Message


def _chunk(delta: SimpleNamespace | None = None, usage: SimpleNamespace | None = None):
    """假流式 chunk：要么带 usage（收尾块，无 choices），要么带 delta。"""
    if usage is not None:
        return SimpleNamespace(usage=usage, choices=[])
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta)])


def _usage(prompt: int, completion: int, total: int) -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _tool_chunk(
    id_: str | None, name: str | None, args: str | None, index: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        index=index, id=id_, function=SimpleNamespace(name=name, arguments=args)
    )


class FakeCompletions:
    """替身：记录收到的 kwargs，返回预设 chunk 流。"""

    def __init__(self, chunks: list) -> None:
        self._chunks = chunks
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self._chunks)


def _client(chunks: list) -> tuple[LLMClient, FakeCompletions]:
    llm = LLMClient(Config())
    fake = FakeCompletions(chunks)
    llm._client.chat.completions.create = fake.create  # type: ignore[method-assign]
    return llm, fake


def test_stream_accumulates_content_and_reasoning() -> None:
    llm, _ = _client(
        [
            _chunk(delta=SimpleNamespace(content="你", reasoning_content="思", tool_calls=None)),
            _chunk(delta=SimpleNamespace(content="好", reasoning_content="考", tool_calls=None)),
            _chunk(usage=_usage(10, 5, 15)),
        ]
    )
    result = llm.chat([Message("user", "hi")])
    assert result.messages[0].content == "你好"
    assert result.messages[0].reasoning_content == "思考"
    assert result.usage is not None and result.usage.total_tokens == 15
    assert result.tool_calls == []


def test_stream_concatenates_tool_call_fragments() -> None:
    llm, _ = _client(
        [
            _chunk(delta=SimpleNamespace(content=None, reasoning_content=None, tool_calls=None)),
            _chunk(delta=SimpleNamespace(
                content=None, reasoning_content=None,
                tool_calls=[_tool_chunk("call_1", "shell", "")])),
            _chunk(delta=SimpleNamespace(
                content=None, reasoning_content=None,
                tool_calls=[_tool_chunk(None, None, '{"command')])),
            _chunk(delta=SimpleNamespace(
                content=None, reasoning_content=None,
                tool_calls=[_tool_chunk(None, None, '": "ls"}')])),
            _chunk(usage=_usage(10, 5, 15)),
        ]
    )
    result = llm.chat([Message("user", "hi")])
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "shell"
    assert tc.arguments == '{"command": "ls"}'
    # assistant 消息携带完整 tool_calls，供持久化与回传
    assert result.messages[0].tool_calls == [tc]


def test_stream_reads_reasoning_from_model_extra() -> None:
    """openai SDK 3.x 的 ChoiceDelta 不声明 reasoning_content，只在 model_extra。"""
    llm, _ = _client(
        [
            _chunk(delta=SimpleNamespace(
                content=None, reasoning_content=None, tool_calls=None,
                model_extra={"reasoning_content": "思考"})),
            _chunk(usage=_usage(10, 5, 15)),
        ]
    )
    result = llm.chat([Message("user", "hi")])
    assert result.messages[0].reasoning_content == "思考"


def test_stream_passes_reasoning_back_in_next_call() -> None:
    """第二轮：历史里的 assistant 消息带 reasoning_content，to_openai 原样回传。"""
    llm, fake = _client([_chunk(usage=_usage(10, 5, 15))])
    hist = [Message("assistant", "hi", reasoning_content="思考")]
    llm.chat(hist)
    sent = fake.kwargs["messages"]
    assert sent[0]["role"] == "assistant"
    assert sent[0]["reasoning_content"] == "思考"


# -- M5：交错分片按 id 防御合并 -------------------------------------------------


def test_stream_merges_interleaved_tool_calls() -> None:
    """两个工具调用的参数分片交错：按 .index 槽正确归位，顺序保持。"""
    llm, _ = _client(
        [
            _chunk(delta=SimpleNamespace(content=None, reasoning_content=None,
                                         tool_calls=[_tool_chunk("call_a", "read_file", '{"path', index=0)])),
            _chunk(delta=SimpleNamespace(content=None, reasoning_content=None,
                                         tool_calls=[_tool_chunk("call_b", "shell", '{"cmd', index=1)])),
            _chunk(delta=SimpleNamespace(content=None, reasoning_content=None,
                                         tool_calls=[_tool_chunk(None, None, '": 1"}', index=0)])),
            _chunk(delta=SimpleNamespace(content=None, reasoning_content=None,
                                         tool_calls=[_tool_chunk(None, None, '": "ls"}', index=1)])),
            _chunk(usage=_usage(10, 5, 15)),
        ]
    )
    result = llm.chat([Message("user", "hi")])
    assert [(tc.id, tc.name, tc.arguments) for tc in result.tool_calls] == [
        ("call_a", "read_file", '{"path": 1"}'),
        ("call_b", "shell", '{"cmd": "ls"}'),
    ]


def test_stream_resets_slot_on_id_change() -> None:
    """同一 index 出现新 id（异常流）：视为新工具调用，不把新旧参数拼一起。"""
    llm, _ = _client(
        [
            _chunk(delta=SimpleNamespace(content=None, reasoning_content=None,
                                         tool_calls=[_tool_chunk("call_1", "shell", '{"a": 1}')])),
            _chunk(delta=SimpleNamespace(content=None, reasoning_content=None,
                                         tool_calls=[_tool_chunk("call_2", "shell", '{"b": 2}')])),
            _chunk(usage=_usage(10, 5, 15)),
        ]
    )
    result = llm.chat([Message("user", "hi")])
    assert [(tc.id, tc.arguments) for tc in result.tool_calls] == [
        ("call_1", '{"a": 1}'),
        ("call_2", '{"b": 2}'),
    ]


def test_stream_reasoning_callback() -> None:
    """on_reasoning 回调逐分片收到思考内容。"""
    got: list[str] = []
    llm, _ = _client(
        [
            _chunk(delta=SimpleNamespace(content="你", reasoning_content="思", tool_calls=None)),
            _chunk(delta=SimpleNamespace(content="好", reasoning_content="考", tool_calls=None)),
            _chunk(usage=_usage(10, 5, 15)),
        ]
    )
    llm.chat([Message("user", "hi")], on_reasoning=got.append)
    assert got == ["思", "考"]


# -- M5：可重试错误退避 ---------------------------------------------------------


def _client_with_create(fake_create) -> LLMClient:
    llm = LLMClient(Config(), max_retries=2)
    llm._client.chat.completions.create = fake_create  # type: ignore[method-assign]
    return llm


def test_retry_succeeds_after_transient_errors(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr("vgent.llm._RETRYABLE", (RuntimeError,))
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return iter([_chunk(usage=_usage(10, 5, 15))])

    result = _client_with_create(flaky).chat([Message("user", "hi")])
    assert calls["n"] == 3
    assert result.usage is not None and result.usage.total_tokens == 15


def test_retry_gives_up_after_max_retries(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr("vgent.llm._RETRYABLE", (RuntimeError,))
    calls = {"n": 0}

    def always_fail(**kwargs):
        calls["n"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _client_with_create(always_fail).chat([Message("user", "hi")])
    assert calls["n"] == 3  # 1 次 + 2 次重试


def test_no_retry_on_non_retryable_error(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = {"n": 0}

    def bad(**kwargs):
        calls["n"] += 1
        raise ValueError("确定性问题")

    with pytest.raises(ValueError):
        _client_with_create(bad).chat([Message("user", "hi")])
    assert calls["n"] == 1
