"""⑥ 工具层：JSON Schema 定义 + dispatcher；内置 shell / read_file / write_file / search。

契约 v0.1：`ToolRegistry.register(tool, handler)` / `schemas() -> list[dict]` /
`execute(name, args) -> str`。permission 三档（read/write/exec）由权限系统消费（决策 7）。
M5 补齐决策 5 的 v1 工具面：write_file（write 档）+ search（read 档，递归正则）。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import winreg  # Windows: Git for Windows 注册表 InstallPath（非标准安装路径）
except ImportError:  # 非 Windows
    winreg = None

Permission = Literal["read", "write", "exec"]

# 工具输出硬上限：防病态输出撑爆上下文（M3 的剪枝是精细层，这里是安全底）
_OUTPUT_CAP = 10_000


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: dict  # JSON Schema
    permission: Permission


@dataclass
class Tool:
    schema: ToolSchema
    handler: Callable[[dict], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: ToolSchema, handler: Callable[[dict], str]) -> None:
        self._tools[tool.name] = Tool(tool, handler)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        """转成 OpenAI chat/completions 的 tools 参数格式（决策 9：原生 tool calling）。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.schema.name,
                    "description": t.schema.description,
                    "parameters": t.schema.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, args: dict) -> str:
        return self._tools[name].handler(args)

    def filter_denied(self, denied: list[str]) -> None:
        """P10：deny 规则的工具从注册表移除——schemas() 不再暴露，模型看不到。

        由 cli/web 在接线时调用（内置工具 + MCP 工具一起过滤）；check() 的
        DENIED 分支仍保留作直接 execute 的兜底。
        """
        for name in denied:
            self._tools.pop(name, None)


# -- 内置工具 -------------------------------------------------------------


# Git Bash 固定候选路径（Git 装在非标准位置时，靠注册表/PATH 里的 git 兜底）
_SHELL_CANDIDATES = (
    Path(r"C:\Program Files\Git\bin\bash.exe"),
    Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
    Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
    Path(r"C:\msys64\usr\bin\bash.exe"),
)


def _git_roots_from_registry() -> list[Path]:
    """Git for Windows 注册表 InstallPath（HKLM 优先，HKCU 兜底）。"""
    if winreg is None:
        return []
    roots: list[Path] = []
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, r"SOFTWARE\GitForWindows") as key:
                value, _ = winreg.QueryValueEx(key, "InstallPath")
            roots.append(Path(value))
        except OSError:
            continue
    return roots


def _git_roots_from_git_path() -> list[Path]:
    """从 PATH 里的 git 推断 Git 根目录（git 可能在 cmd/ 或 mingw64/bin/ 下）。"""
    git = shutil.which("git")
    if not git:
        return []
    p = Path(git).resolve()
    roots: list[Path] = []
    for _ in range(3):
        p = p.parent
        roots.append(p)
        if p.name.lower() == "git":
            break
    return roots


def _resolve_shell() -> str | None:
    """优先 Git Bash（Windows 决策 4：shell 层适配），其次 PATH 里的 bash/sh。

    候选顺序：固定路径 → 注册表 InstallPath → PATH 里 git 推断的根 → PATH 里的 bash/sh。
    """
    candidates = list(_SHELL_CANDIDATES)
    for root in [*_git_roots_from_registry(), *_git_roots_from_git_path()]:
        for rel in (Path("usr/bin/bash.exe"), Path("bin/bash.exe")):
            candidates.append(root / rel)
    for c in candidates:
        if c.exists():
            return str(c)
    for name in ("bash", "sh"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _cap_output(text: str) -> str:
    if len(text) <= _OUTPUT_CAP:
        return text
    return text[:_OUTPUT_CAP] + f"\n...(输出已截断，共 {len(text)} 字符)"


def _shell_handler(args: dict) -> str:
    cmd = str(args.get("command", "")).strip()
    if not cmd:
        return "错误：缺少 command 参数"
    try:
        timeout = int(args.get("timeout", 120) or 120)
    except (TypeError, ValueError):
        timeout = 120
    shell = _resolve_shell()
    if shell is None:
        return "错误：未找到可用 shell（bash/sh）"
    try:
        # 显式 UTF-8 + errors=replace：中文 Windows 默认 GBK 解码会崩（真机首跑踩坑），
        # 输出含 UTF-8/非法字节时不能丢输出
        proc = subprocess.run(
            [shell, "-lc", cmd],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial = _cap_output((exc.stdout or "").rstrip())
        return f"命令超时（>{timeout}s）\n部分输出：\n{partial}"
    except OSError as exc:
        return f"执行失败：{exc}"
    out = ((proc.stdout or "") + (proc.stderr or "")).rstrip()
    if not out:
        return f"exit {proc.returncode}（无输出）"
    return f"exit {proc.returncode}\n{_cap_output(out)}"


def _read_file_handler(args: dict) -> str:
    raw = args.get("path")
    if not raw:
        return "错误：缺少 path 参数"
    path = Path(str(raw))
    try:
        offset = max(1, int(args.get("offset", 1) or 1))
    except (TypeError, ValueError):
        offset = 1
    try:
        limit = max(0, int(args.get("limit", 0) or 0))
    except (TypeError, ValueError):
        limit = 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"读取失败：{exc}"
    lines = text.splitlines()
    start = offset - 1
    if start >= len(lines):
        return "(offset 超出文件行数)"
    chunk = lines[start:] if limit <= 0 else lines[start : start + limit]
    return "\n".join(f"{i:>6}\t{ln}" for i, ln in enumerate(chunk, start=start + 1))


# M5：搜索时跳过这些目录（避免拖慢 / 刷屏）；.zcode/.cache 为真机首跑补（会搜出 agent 会话产物）
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache", ".ruff_cache", "dist", "build", ".idea", ".vscode", ".hg", ".svn", ".zcode", ".cache"}
_SEARCH_LINE_CAP = 200  # 匹配行单行截断
_SEARCH_RESULT_CAP = 100  # 结果条数上限


def _search_handler(args: dict) -> str:
    """递归正则搜索文本文件：`file:lineno: 内容`；默认当前目录。"""
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        return "错误：缺少 pattern 参数"
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"正则错误：{exc}"
    raw_path = str(args.get("path", ".") or ".")
    path = Path(raw_path)
    try:
        limit = max(1, int(args.get("limit", _SEARCH_RESULT_CAP) or _SEARCH_RESULT_CAP))
    except (TypeError, ValueError):
        limit = _SEARCH_RESULT_CAP

    results: list[str] = []
    files: list[Path]
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = []
        for root, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            files.extend(Path(root) / n for n in names)
    else:
        return f"路径不存在：{path}"

    for f in files:
        if len(results) >= limit:
            break
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                shown = line.strip() if len(line) <= _SEARCH_LINE_CAP else line.strip()[:_SEARCH_LINE_CAP] + "…"
                results.append(f"{f}:{lineno}: {shown}")
                if len(results) >= limit:
                    break
    if not results:
        return f"未找到匹配 pattern={pattern!r}（{path}）"
    body = "\n".join(results)
    if len(body) > _OUTPUT_CAP:
        body = body[:_OUTPUT_CAP] + f"\n...(输出已截断，共 {len(results)} 条)"
    return body


def _write_file_handler(args: dict) -> str:
    """写文件（覆盖或追加），自动建目录。write 档，执行前需权限确认。"""
    raw = str(args.get("path", "")).strip()
    if not raw:
        return "错误：缺少 path 参数"
    content = str(args.get("content", ""))
    mode = str(args.get("mode", "overwrite") or "overwrite").strip().lower()
    if mode not in ("overwrite", "append"):
        return f"错误：mode 仅支持 overwrite / append，收到 {mode!r}"
    path = Path(raw)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if mode == "append" else "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as exc:
        return f"写入失败：{exc}"
    return f"已{mode} {len(content)} 字符 → {path}"


def _edit_file_handler(args: dict) -> str:
    """P1：手术式编辑——精确字符串替换（claude FileEditTool 简化版，决策 9 防御式）。

    - old_string 必须唯一匹配（replace_all=false），多义时报错回喂模型要求更多上下文；
    - old_string == new_string 拒绝（空操作）；
    - 未找到 / 读取失败时不写文件，错误文本回喂模型自纠正。
    """
    raw = str(args.get("path", "")).strip()
    if not raw:
        return "错误：缺少 path 参数"
    old = str(args.get("old_string", ""))
    new = str(args.get("new_string", ""))
    if not old:
        return "错误：缺少 old_string 参数"
    if old == new:
        return "错误：old_string 与 new_string 相同（空操作），请检查替换目标"
    ra = args.get("replace_all", False)
    if isinstance(ra, str):
        replace_all = ra.strip().lower() in ("1", "true", "yes")
    else:
        replace_all = bool(ra)
    path = Path(raw)
    try:
        # 严格解码（评审 F2）：errors="replace" 读 + 写回 = 有损往返，非 UTF-8 文件
        # （如 GBK）编辑一次原文就永久损坏——拒绝编辑并回喂模型。
        # read_text 默认 errors="strict" 且带通用换行转换（与写回 write_text 配对）
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"读取失败：{exc}"
    except UnicodeDecodeError:
        return (
            "错误：文件含非 UTF-8 字节，已拒绝编辑（防止有损写回损坏原文）。"
            "请先用 shell 工具确认文件编码/内容，或让用户处理后重试"
        )
    count = text.count(old)
    if count == 0:
        return (
            f"错误：文件中未找到要替换的文本（{len(old)} 字符）。"
            "请确认 old_string 精确匹配当前文件内容"
        )
    if count > 1 and not replace_all:
        return (
            f"错误：old_string 在文件中出现 {count} 次，替换目标不明确。"
            "请提供更多上下文使 old_string 唯一匹配，或设置 replace_all=true"
        )
    try:
        path.write_text(text.replace(old, new), encoding="utf-8")
    except OSError as exc:
        return f"写入失败：{exc}"
    return f"已替换 {count} 处 → {path}"


def default_tools() -> ToolRegistry:
    """M5 内置工具：shell（exec 档）+ read_file / search（read 档）+ write_file（write 档）。"""
    reg = ToolRegistry()
    reg.register(
        ToolSchema(
            name="shell",
            description=(
                "在本地 shell 中执行命令（Windows 上为 Git Bash）。"
                "用于运行脚本、查看目录/进程、批量操作等；相对路径基于 vgent 启动目录。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的完整命令"},
                    "timeout": {"type": "integer", "description": "超时秒数，默认 120"},
                },
                "required": ["command"],
            },
            permission="exec",
        ),
        _shell_handler,
    )
    reg.register(
        ToolSchema(
            name="read_file",
            description=(
                "读取文件内容（UTF-8），带行号。offset 起始行（默认 1），"
                "limit 最多读取行数（默认全部）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对或绝对）"},
                    "offset": {"type": "integer", "description": "起始行号，默认 1"},
                    "limit": {"type": "integer", "description": "最多读取行数，默认全部"},
                },
                "required": ["path"],
            },
            permission="read",
        ),
        _read_file_handler,
    )
    reg.register(
        ToolSchema(
            name="search",
            description=(
                "在文本文件中递归搜索正则表达式，输出 file:lineno: 内容。"
                "自动跳过 .git/node_modules/.venv 等目录；limit 为结果条数上限（默认 100）。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "正则表达式"},
                    "path": {"type": "string", "description": "文件或目录（默认当前目录 .）"},
                    "limit": {"type": "integer", "description": "最多返回条数，默认 100"},
                },
                "required": ["pattern"],
            },
            permission="read",
        ),
        _search_handler,
    )
    reg.register(
        ToolSchema(
            name="write_file",
            description=(
                "写入文件（UTF-8）。mode=overwrite 覆盖（默认）/ append 追加；自动创建父目录。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对或绝对）"},
                    "content": {"type": "string", "description": "要写入的完整内容"},
                    "mode": {"type": "string", "description": "overwrite（默认）| append"},
                },
                "required": ["path", "content"],
            },
            permission="write",
        ),
        _write_file_handler,
    )
    reg.register(
        ToolSchema(
            name="edit_file",
            description=(
                "手术式编辑文件（P1）：精确字符串匹配替换，适合局部修改大文件、避免整文件重写。"
                "old_string 必须唯一匹配（出现多次时报错，需更多上下文或 replace_all=true）；"
                "old_string 与 new_string 相同会被拒绝；未找到会报错且不写文件。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对或绝对）"},
                    "old_string": {"type": "string", "description": "要替换的精确原文（需唯一匹配）"},
                    "new_string": {"type": "string", "description": "替换后的内容"},
                    "replace_all": {
                        "type": "boolean",
                        "description": "替换所有匹配（默认 false）",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
            permission="write",
        ),
        _edit_file_handler,
    )
    return reg
