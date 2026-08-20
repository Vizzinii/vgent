"""消息模型（②loop / ③LLM / ④存储 共用）——契约 v0.1。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON 原文；容忍坏参数（决策 9）


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    reasoning_content: str | None = None  # DeepSeek 思考模式产物；必须原样回传，否则 API 400
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        """转成 OpenAI chat/completions 消息格式。"""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
