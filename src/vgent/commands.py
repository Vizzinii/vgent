"""M10：外部命令扩展（zcode 式 slash 命令）。

约定：`~/.vgent/commands/<name>.py` 里定义 `run(ctx, args: str) -> str`，
返回文本由 REPL 打印；命令名即文件名（需为合法标识符）。
坏文件/无 run() 跳过并记日志，不阻塞启动。内置命令优先级高于外部命令。
"""
from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from pathlib import Path

from vgent.agent import SessionContext

Command = Callable[[SessionContext, str], str]

_LOGGER = logging.getLogger("vgent")


def load_commands(commands_dir: Path) -> dict[str, Command]:
    """扫描 commands_dir 下的 *.py，导入模块的 run()。"""
    commands: dict[str, Command] = {}
    if not commands_dir.is_dir():
        return commands
    for f in sorted(commands_dir.glob("*.py")):
        name = f.stem
        if not name.isidentifier():
            _LOGGER.warning("外部命令 %s 文件名不是合法标识符（已跳过）", name)
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"vgent_ext_{name}", f)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            run = getattr(module, "run", None)
        except Exception:  # noqa: BLE001 — 坏命令文件跳过，不阻塞启动
            _LOGGER.warning("外部命令 %s 加载失败（已跳过）", name)
            continue
        if callable(run):
            commands[name] = run
        else:
            _LOGGER.warning("外部命令 %s 没有可调用的 run(ctx, args)（已跳过）", name)
    return commands
