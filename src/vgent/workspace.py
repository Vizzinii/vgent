"""M10：工作区级指令（AGENTS.md / CLAUDE.md）发现与读取。

zcode / Claude Code 惯例：项目根放 AGENTS.md（或 CLAUDE.md），启动时读取，
作为 system 指令注入首个 LLM 调用（不落库，与 cwd_anchor 同模式）。
从当前工作目录向上找最近的一份；内容超长截断，防撑爆首个上下文。
"""
from __future__ import annotations

from pathlib import Path

_INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md")
_INSTRUCTIONS_CAP = 8_000  # 字符上限
_MAX_PARENT_DEPTH = 8


def find_instructions(cwd: Path | str) -> tuple[str, str] | None:
    """从 cwd 向上找最近的指令文件，返回 (文件名, 内容)；找不到返回 None。"""
    p = Path(cwd).resolve()
    for _ in range(_MAX_PARENT_DEPTH):
        for name in _INSTRUCTION_FILES:
            f = p / name
            if f.is_file():
                try:
                    text = f.read_text(encoding="utf-8", errors="replace").strip()
                except OSError:
                    continue
                if not text:
                    continue
                if len(text) > _INSTRUCTIONS_CAP:
                    text = text[:_INSTRUCTIONS_CAP] + "…（指令文件过长，已截断）"
                return name, text
        if p.parent == p:
            break
        p = p.parent
    return None
