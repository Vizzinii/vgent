r"""⑤ 上下文引擎 —— ContextEngine（M3，决策 8 综合方案）。

蓝本：hermes-agent ContextCompressor（参考库，只读参照）；
切点对齐回合边界取自 ag2；硬下限兜底取自 OpenManus。

契约 v0.1：
  ① update_from_response(usage)                每轮 chat 后上报（③→⑤，usage 校准 _tokens）
  ② should_compress() -> bool                  发请求前问；高水位（默认 75%）触发
    compress(messages) -> messages             只动发送列表，SQLite 全量历史不动
  ⑤ prune_tool_results_only(messages) -> (messages, n)  低水位免费剪枝

策略（决策 8，v1 内置 TailWindow + Summarize）：
  - 低水位（prune_percent=0.30）：工具结果压成一行摘要 + 清孤儿 tool 对；
    受保护的最新消息按条数（hermes 经验：token 预算在 1M 窗口会保护全部、剪不掉）。
  - 高水位（threshold_percent=0.75）：保护头部（首条）+ 尾部（tail_token_budget），
    中间压缩并插入标记消息；丢弃切点对齐回合边界，不切分 [assistant(tool_calls)+tool*] 对；
    仍超窗时硬下限兜底（从最旧丢起，保留最后一条）。
  - 中间压缩策略：TailWindow（默认，零成本，丢中间）| Summarize（M4：LLM 摘要中间段，
    需注入 summarizer；/compact 手动触发 + config 可配自动）。摘要失败自动退回 TailWindow。
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from vgent.config import ContextConfig
from vgent.messages import Message, Usage

# P4（claude BASE_COMPACT_PROMPT 简化版）：压缩/记忆摘要的结构化模板
COMPACT_PROMPT = (
    "你是会话压缩器。把下面的对话历史压缩成详细摘要，供后续对话继续使用。\n"
    "先输出 <analysis> 块组织你的思考（草稿，不进入最终上下文），"
    "再输出 <summary> 块作为最终摘要。\n"
    "摘要必须覆盖：\n"
    "1. 未完成的任务与下一步计划；\n"
    "2. 已做出的关键决策与采用的技术方案；\n"
    "3. 关键事实（文件路径、命令、数据、代码模式）；\n"
    "4. 用户明确的安全约束或偏好（如涉及，原样保留，不得改写）。\n"
    "只输出 <analysis> 和 <summary> 两个块。"
)


def extract_summary(text: str) -> str:
    """从结构化压缩输出里取 <summary> 块（analysis 是草稿不进上下文）；无块则原文。"""
    m = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text

# 工具结果摘要：首行截断上限
_PRUNE_SUMMARY_CAP = 200
# 低水位剪枝保护的最新消息条数（hermes protect_last_n 思路）
_PRUNE_PROTECT_LAST_N = 6


@dataclass
class ContextEngine:
    context_length: int = 1_000_000
    cfg: ContextConfig = field(default_factory=ContextConfig)
    # Summarize 策略用的 LLM 摘要器（cli 注入；None 时 summarize 退回 TailWindow）
    summarizer: Callable[[list[Message]], str] | None = None
    # 当前发送列表的 token 估算：API usage 校准 + 本地启发式（决策 8：usage 计 token）
    _tokens: int = field(default=0, init=False, repr=False)
    compression_count: int = field(default=0, init=False, repr=False)
    # /compact 后的压缩列表：作为后续 run_turn 的发送底稿（SQLite 全量历史不动）
    compacted: list[Message] | None = field(default=None, init=False, repr=False)

    # -- 契约① usage 上报 -----------------------------------------------------

    def update_from_response(self, usage: Usage | None) -> None:
        """每轮 chat 后调用；用 API 的 total_tokens 校准当前列表规模。"""
        if usage is not None:
            self._tokens = usage.total_tokens

    # -- 契约② 高水位 ---------------------------------------------------------

    def should_compress(self) -> bool:
        return self._tokens > self._watermark(self.cfg.threshold_percent)

    def compress(
        self,
        messages: list[Message],
        strategy: str | None = None,
        force: bool = False,
    ) -> list[Message]:
        """压缩中间段（保护头部 + 尾部预算，对齐回合边界）。

        strategy：缺省取 cfg.compact_strategy（"tail" 零成本 | "summarize" LLM 摘要）；
        force=True 跳过水位检查（/compact 手动触发用）。
        """
        self._sync_estimate(messages)
        if (not force and not self.should_compress()) or len(messages) <= 1:
            return messages

        # 尾部：从后往前累计，直到预算耗尽；start = 尾部第一条的下标
        start = len(messages)
        tail_est = 0
        while start > 1 and tail_est < self.cfg.tail_token_budget:
            start -= 1
            tail_est += self._estimate_tokens([messages[start]])
        # 对齐回合边界：尾部起点若是 tool 结果，连同其 assistant tool_calls 一起保留
        while start > 1 and messages[start].role == "tool":
            start -= 1
        dropped = start - 1  # messages[1..start-1] 为中间段（头部之外、尾部之前）
        if dropped <= 0:
            return messages

        strategy = strategy or self.cfg.compact_strategy
        marker = (
            self._summarize(messages[1:start], dropped)
            if strategy == "summarize"
            else self._marker(dropped)
        )
        result = [messages[0], marker] + messages[start:]
        result = self._cleanup_tool_pairs(result)
        # 硬下限兜底（OpenManus）：极端超长仍丢最旧，保留最后一条
        while self._estimate_tokens(result) > self.context_length and len(result) > 1:
            result.pop(0)
        self.compression_count += 1
        return result

    # -- 契约⑤ 低水位免费剪枝 -------------------------------------------------

    def prune_tool_results_only(
        self, messages: list[Message]
    ) -> tuple[list[Message], int]:
        """工具结果一行摘要 + 清孤儿 tool 对；低于低水位或没可剪的则原样返回。"""
        self._sync_estimate(messages)
        if self._tokens <= self._watermark(self.cfg.prune_percent):
            return messages, 0
        # 受保护的最新消息按条数，只剪更早的工具结果
        cut = len(messages) - _PRUNE_PROTECT_LAST_N
        if cut <= 0:
            return messages, 0
        pruned = list(messages)
        n = 0
        for i in range(cut):
            m = pruned[i]
            if m.role == "tool" and len(m.content) > _PRUNE_SUMMARY_CAP:
                pruned[i] = Message(
                    "tool",
                    self._one_line_summary(m.content),
                    tool_call_id=m.tool_call_id,
                )
                n += 1
        if n == 0:
            return messages, 0
        return self._cleanup_tool_pairs(pruned), n

    # -- 内部 ---------------------------------------------------------------

    def _sync_estimate(self, messages: list[Message]) -> None:
        self._tokens = self._estimate_tokens(messages)

    @staticmethod
    def _estimate_tokens(messages: list[Message]) -> int:
        """本地启发式估算（中文按 ~3 字符/token 偏保守），供剪枝/压缩决策。"""
        total = 0
        for m in messages:
            total += 4  # role 与格式开销
            total += len(m.content) // 3
            if m.reasoning_content:
                total += len(m.reasoning_content) // 3
            if m.tool_calls:
                for tc in m.tool_calls:
                    total += 4 + len(tc.arguments) // 3
        return total

    def _watermark(self, percent: float) -> int:
        return max(1, min(int(self.context_length * percent), self.context_length - 1))

    @staticmethod
    def _marker(dropped: int) -> Message:
        return Message(
            "system",
            f"[vgent] 中间 {dropped} 条历史已由 TailWindow 压缩，保留开头与最近内容",
        )

    def _summarize(self, middle: list[Message], dropped: int) -> Message:
        """Summarize 策略：LLM 把中间段压成摘要；无 summarizer 或失败退回 TailWindow 标记。"""
        if self.summarizer is None:
            return self._marker(dropped)
        try:
            text = (self.summarizer(middle) or "").strip()
        except Exception:  # noqa: BLE001 — 摘要失败不阻断压缩
            return self._marker(dropped)
        if not text:
            return self._marker(dropped)
        return Message("system", f"【历史摘要（原 {dropped} 条）】{text}")

    @staticmethod
    def _one_line_summary(text: str) -> str:
        line = text.splitlines()[0].strip() if text.splitlines() else ""
        if len(line) > _PRUNE_SUMMARY_CAP:
            line = line[: _PRUNE_SUMMARY_CAP] + "…"
        return f"{line}（原 {len(text)} 字符，已摘要）"

    @staticmethod
    def _cleanup_tool_pairs(messages: list[Message]) -> list[Message]:
        """清孤儿 tool 对（防御）：tool 结果必须紧随其 assistant tool_calls，缺配对即清。

        第一遍从前往后：无主 tool 结果丢弃；第二遍：assistant 声明了 tool_call 却
        始终没有结果——摘除其 tool_calls（内容空则整条丢弃）。
        """
        declared: set[str] = set()  # 所有 assistant 声明过的 tool_call id
        received: set[str] = set()  # 保留下来且收到结果的 id
        active: set[str] = set()  # 当前可接受结果的 id（最近一次 assistant 声明）
        out: list[Message] = []
        for m in messages:
            if m.role == "assistant" and m.tool_calls:
                ids = {tc.id for tc in m.tool_calls}
                declared |= ids
                active = set(ids)
                out.append(m)
            elif m.role == "tool":
                if m.tool_call_id in active:
                    active.discard(m.tool_call_id)
                    received.add(m.tool_call_id)
                    out.append(m)
                # 无主 tool 结果：丢弃
            else:
                active = set()
                out.append(m)
        dangling = declared - received
        if not dangling:
            return out
        fixed: list[Message] = []
        for m in out:
            if m.role == "assistant" and m.tool_calls and any(
                tc.id in dangling for tc in m.tool_calls
            ):
                rest = [tc for tc in m.tool_calls if tc.id not in dangling]
                if rest:
                    fixed.append(Message(m.role, m.content, m.reasoning_content, rest, m.tool_call_id))
                elif m.content:
                    fixed.append(Message(m.role, m.content, m.reasoning_content, None, m.tool_call_id))
                # 内容为空：整条丢弃
            else:
                fixed.append(m)
        return fixed
