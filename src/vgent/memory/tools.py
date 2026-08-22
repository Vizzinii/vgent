"""M12-C 记忆检索工具：memory_read / memory_grep（read 档，模型可主动查）。

思路来源（只学思路不抄代码）：claude memdir / codex memories 的检索工具——
system 只注入 memory_summary 短总览，细节靠工具按需读；memory_read 拒绝重读
memory_summary.md（已在 system，浪费 token）。

handler 用**闭包工厂**注入 data_home/workspace（工具 handler 契约保持纯
`handler(args)`，R2）；由 cli/web 接线时注册（与 MCP 工具注册同模式）。
"""
from __future__ import annotations

from pathlib import Path

from vgent.memory.store import SUMMARY_NAME, MemoryFileStore
from vgent.tools import Tool, ToolSchema

_READ_DESC = (
    "Read a file under the project long-term memory directory "
    "(MEMORY.md, rollout_summaries/..., raw_memories.md). "
    "Do NOT re-read memory_summary.md — it is already in the system prompt. "
    "Path is relative to the memories root."
)
_GREP_DESC = (
    "Search project long-term memory (MEMORY.md and recent rollouts) by "
    "space-separated keywords (AND). Prefer this before opening large MEMORY.md."
)


def make_memory_tools(data_home: Path, workspace: Path) -> list[Tool]:
    """返回 memory_read / memory_grep 两个只读工具（data_dir + workspace 闭包注入）。"""
    store = MemoryFileStore(data_home, workspace)

    def _read(args: dict) -> str:
        path = str(args.get("path", "")).strip()
        if not path:
            return "memory_read error: path is required"
        # summary 已在 system，再读是浪费 token（学 claude）
        if path.strip().replace("\\", "/").endswith(SUMMARY_NAME):
            return (
                "memory_read error: memory_summary.md is already in the system prompt; "
                "use MEMORY.md or rollout_summaries instead."
            )
        try:
            return store.read_rel(path.strip())
        except PermissionError as exc:
            return f"memory_read error: {exc}"
        except FileNotFoundError:
            return f"memory_read error: not found: {path}"
        except OSError as exc:
            return f"memory_read error: {exc}"

    def _grep(args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "memory_grep error: query is required"
        return store.grep(query.strip())

    return [
        Tool(
            ToolSchema(
                name="memory_read",
                description=_READ_DESC,
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": 'Relative path e.g. "MEMORY.md" or "rollout_summaries/xxx.md"',
                        },
                    },
                    "required": ["path"],
                },
                permission="read",
            ),
            _read,
        ),
        Tool(
            ToolSchema(
                name="memory_grep",
                description=_GREP_DESC,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Space-separated keywords (AND)",
                        },
                    },
                    "required": ["query"],
                },
                permission="read",
            ),
            _grep,
        ),
    ]
