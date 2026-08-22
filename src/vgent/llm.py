"""③ LLM Provider 客户端 —— openai SDK（sync，base_url 指 DeepSeek）。

契约④：`chat(messages, tools, on_delta=None, on_reasoning=None) -> ChatResult`
（流式，on_delta → ① 渲染；on_reasoning → 思考过程渲染，M5 可选）。
usage 取自流末尾的 include_usage 块，供契约①（M3 接 ContextEngine）使用。

M5：
- 可重试错误（429/5xx/连接/超时）指数退避重试；SDK 内建重试关闭（max_retries=0）避免叠加；
  401/400 等确定性错误不重试，直接抛给 REPL 兜底。
- tool_calls 分片按「index 槽 + id 校验」合并：同一 index 出现新 id 视为新工具调用
  （防御交错分片 / index 复用的异常流）。
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from vgent.config import Config
from vgent.messages import Message, ToolCall, Usage

# 重试退避（秒）：第 n 次重试前等待 _RETRY_BACKOFF[n-1]
_RETRY_BACKOFF = (1.0, 2.0, 4.0)
# 可重试错误：限流 / 服务端 5xx / 连接与超时（网络瞬断）
_RETRYABLE = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


@dataclass
class ChatResult:
    messages: list[Message]  # assistant 消息（含 tool_calls），供持久化
    usage: Usage | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient:
    def __init__(self, cfg: Config, max_retries: int = 3) -> None:
        self.cfg = cfg
        self.max_retries = max_retries
        # SDK 构造时要求非空 key；未配置时用占位 key，让失败发生在请求时（401），
        # 由 REPL 兜底优雅报错，而不是启动即崩。max_retries=0：重试由本类自建。
        # timeout=120（评审 F15）：SDK 默认 600s，卡死请求要等 10 分钟才进重试。
        self._client = OpenAI(
            base_url=cfg.provider.base_url,
            api_key=cfg.api_key_resolved() or "sk-vgent-missing-key",
            max_retries=0,
            timeout=120.0,
        )

    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> ChatResult:
        """流式对话。可重试错误退避重试；确定性问题直接抛（REPL 兜底显示）。"""
        kwargs: dict = {
            "model": self.cfg.provider.model,
            "messages": [m.to_openai() for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        attempt = 0
        while True:
            try:
                return self._chat_once(kwargs, on_delta, on_reasoning)
            except _RETRYABLE:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                # 评审 F13：失败前可能已流出部分增量，重试会从头再流——发提示防用户困惑
                if on_delta:
                    on_delta("\n（网络错误，重试中，输出可能重复……）\n")
                delay = _RETRY_BACKOFF[min(attempt - 1, len(_RETRY_BACKOFF) - 1)]
                time.sleep(delay)

    def _chat_once(
        self,
        kwargs: dict,
        on_delta: Callable[[str], None] | None,
        on_reasoning: Callable[[str], None] | None,
    ) -> ChatResult:
        stream = self._client.chat.completions.create(**kwargs)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_acc: dict[str, dict[str, str]] = {}
        slot_index: dict[int, str] = {}
        usage: Usage | None = None
        for chunk in stream:
            if chunk.usage:
                usage = Usage(
                    chunk.usage.prompt_tokens,
                    chunk.usage.completion_tokens,
                    chunk.usage.total_tokens,
                )
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                if on_delta:
                    on_delta(delta.content)
            # openai SDK 3.x 的 ChoiceDelta 不声明 reasoning_content，只进 model_extra
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning is None:
                reasoning = (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning)
                if on_reasoning:
                    on_reasoning(reasoning)
            # tool_calls 合并：有 id 按 id 归槽（保序），无 id 续片按 index 归属当前槽；
            # 同一 index 出现新 id 视为新工具调用（防御交错/复用异常流）。
            # 流式分片的 .index 字段才是消息里的真实序号（chunk 数组常只有一个元素）。
            for i, tc in enumerate(delta.tool_calls or []):
                idx = getattr(tc, "index", None)
                if idx is None:
                    idx = i
                if tc.id:
                    key = tc.id
                    if key not in tool_acc:
                        tool_acc[key] = {"id": "", "name": "", "arguments": ""}
                        ph = f"idx:{idx}"
                        if ph in tool_acc and slot_index.get(idx) == ph:
                            # 无 id 续片先到（异常流）：占位槽并入 id 槽
                            prev = tool_acc.pop(ph)
                            tool_acc[key]["name"] = prev["name"] + tool_acc[key]["name"]
                            tool_acc[key]["arguments"] = (
                                prev["arguments"] + tool_acc[key]["arguments"]
                            )
                    slot_index[idx] = key
                else:
                    key = slot_index.get(idx)
                    if key is None:
                        key = f"idx:{idx}"
                        tool_acc.setdefault(key, {"id": "", "name": "", "arguments": ""})
                        slot_index[idx] = key
                acc = tool_acc[key]
                if tc.id:
                    acc["id"] = tc.id
                if tc.function and tc.function.name:
                    acc["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    acc["arguments"] += tc.function.arguments

        tool_calls = [
            ToolCall(v["id"], v["name"], v["arguments"]) for v in tool_acc.values()
        ]
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts) or None
        return ChatResult(
            messages=[
                Message(
                    "assistant",
                    content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls or None,
                )
            ],
            usage=usage,
            tool_calls=tool_calls,
        )
