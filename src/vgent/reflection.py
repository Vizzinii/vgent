"""v2 演进 · 反思循环（M7）：失败信号 → 显式 LLM 反思 → 修正动作回喂。

- 失败判定：启发式（保守防误判）——shell 非零退出 / 明确错误关键词；
- 反思：失败后追加一次 LLM 调用，产出「Failure / Action」两行，作为 system 消息
  注入下一轮发送列表（自动反思不落库，一次性引导；恢复会话不保留，属预期）；
- 上限：单轮 run_turn 内最多 MAX_REFLECT_ROUNDS 次，防死循环烧 token；
- 模型不配合（空响应/异常/测试桩无响应）自动回退：错误文本回喂由决策 9 兜底。
"""
from __future__ import annotations

import re
from collections.abc import Callable

from vgent.messages import Message

MAX_REFLECT_ROUNDS = 3

REFLECT_PROMPT = (
    "刚才有工具执行失败了。请只做一次简短反思，输出以下两行（不要执行工具，不要输出计划块）：\n"
    "Failure: <失败原因，一句话>\n"
    "Action: <下一步要执行的修正动作，一句话>\n"
    "若原因不明确或已无法修正，Action 写「无法修正，向用户说明」。"
)

# 失败关键词（保守：只匹配明确错误，避免误判额外烧 token）
_FAILURE_MARKERS = (
    "错误", "失败", "未找到", "拒绝", "超时", "执行出错", "解析失败", "未知工具",
    "Traceback", "FAILED", "failed", "command not found", "No such file",
    "Permission denied",
)
# shell 正常退出（exit 0）时仍算失败的强信号（测试失败等）
_STRONG_ONLY = ("FAILED", "Traceback")


def looks_failed(text: str) -> bool:
    """启发式判断工具结果是否失败（保守：宁可不触发，也不误判）。"""
    if not text:
        return False
    m = re.search(r"exit\s+(\d+)", text)
    if m:
        code = int(m.group(1))
        if code != 0:
            return True
        return any(k in text for k in _STRONG_ONLY)
    return any(k in text for k in _FAILURE_MARKERS)


def reflect(msgs: list[Message], llm: Callable) -> str:
    """调 LLM 反思最近历史（最近 20 条足够，省 token），返回 Failure/Action 文本。

    异常或空响应返回 ""（反思失败不阻断对话，决策 9 的错误回喂兜底）。
    """
    prompt = Message("system", REFLECT_PROMPT)
    try:
        result = llm.chat([prompt, *msgs[-20:]])
        text = (result.messages[0].content or "").strip()
    except Exception:  # noqa: BLE001 — 反思 best-effort，任何失败都静默
        return ""
    return text
