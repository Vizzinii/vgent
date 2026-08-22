"""⑦ 权限/确认系统 —— v1 三档（决策 7，用户拍板）+ P2 规则表。

- read：自动放行；write / exec：需确认；确认时可 sticky 放行（本会话内不再问）
- P2（决策 7 升级）：config.toml `[permissions]` 规则表（allow/ask/deny）——
  check() 先查规则（deny > sticky/allow > ask），未命中回落三档；
  deny 的工具同时从 schemas() 裁剪（P10，模型看不到）；`/allow` 把批准
  持久化写回 config.toml（跨会话记住）。
契约③ 的 `confirm(tool, args) -> bool` 在 M2 细化为三态 ConfirmResult
（APPROVE 一次 / ALWAYS 本会话 sticky / REJECT）；确认交互由 CLI 注入
（rich prompt）；未注入时默认拒绝（headless 安全默认）。
"""
from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from vgent.config import PermissionRules
from vgent.tools import ToolSchema


def _outside_workspace(args: dict) -> bool:
    """评审 F8：write 档工具的目标路径是否越出当前工作区。

    相对路径按 cwd 解析，绝对路径直接判定；解析失败按越界（保守）。
    用途：越界写盘（改 ~/.vgent/config.toml、.git/config 等）无视 sticky/allow
    放行、强制确认——恢复决策 7「权限确认即防线」的原意（y 仍可执行，非沙箱）。
    """
    raw = str(args.get("path", "") or "").strip()
    if not raw:
        return False
    try:
        p = Path(raw)
        resolved = (p if p.is_absolute() else Path.cwd() / p).resolve(strict=False)
        return not resolved.is_relative_to(Path.cwd().resolve())
    except (OSError, ValueError):
        return True


class Approval(str, Enum):
    AUTO = "auto"
    NEED_CONFIRM = "need_confirm"
    DENIED = "denied"


class ConfirmResult(str, Enum):
    APPROVE = "approve"  # 仅这一次
    ALWAYS = "always"  # 本会话内 sticky 放行
    REJECT = "reject"


class PermissionSystem:
    def __init__(
        self,
        confirm: Callable[[ToolSchema, dict], ConfirmResult] | None = None,
        rules: PermissionRules | None = None,
    ) -> None:
        self._confirm = confirm
        self._sticky: set[str] = set()
        self.rules = rules or PermissionRules()  # P2：config.toml [permissions]

    def check(self, tool: ToolSchema, args: dict) -> Approval:
        """工具执行前的检查：P2 规则优先，未命中回落三档。"""
        if tool.name in self.rules.deny:
            return Approval.DENIED
        # 评审 F8：write 档越界路径无视 sticky/allow/ask——强制确认（用户 y 仍可执行）
        if tool.permission == "write" and _outside_workspace(args):
            return Approval.NEED_CONFIRM
        if tool.name in self._sticky or tool.name in self.rules.allow:
            return Approval.AUTO
        if tool.name in self.rules.ask:
            return Approval.NEED_CONFIRM
        if tool.permission == "read":
            return Approval.AUTO
        if tool.permission in ("write", "exec"):
            return Approval.NEED_CONFIRM
        return Approval.DENIED  # 未知档位：默认拒绝

    def confirm(self, tool: ToolSchema, args: dict) -> ConfirmResult:
        """走到确认交互；ALWAYS 时自动 sticky（本会话内）。无交互则拒绝。"""
        # 评审 F8：越界写盘不吃 sticky 短路——check 已强制 NEED_CONFIRM，这里必须真问
        if tool.name in self._sticky and not (
            tool.permission == "write" and _outside_workspace(args)
        ):
            return ConfirmResult.APPROVE
        if self._confirm is None:
            return ConfirmResult.REJECT
        result = self._confirm(tool, args)
        if result is ConfirmResult.ALWAYS:
            self._sticky.add(tool.name)
        return result

    def approve_sticky(self, tool_name: str) -> None:
        """外部直接 sticky（供 /allow 与测试使用）。"""
        self._sticky.add(tool_name)


def persist_allow(data_dir: Path, tool_name: str) -> bool:
    """P2：把工具名写进 config.toml 的 [permissions].allow（跨会话记住）。

    TOML 无标准序列化器（tomllib 只读）：文本层面替换 [permissions] 段，
    保留该段其他键（ask/deny）与文件中其余内容；段顺序无关紧要。
    文件不存在时不创建（避免生成残缺配置），返回 False（本会话 sticky 仍生效）。
    """
    path = data_dir / "config.toml"
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
        data = tomllib.loads(text)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    perms = data.get("permissions", {})
    if not isinstance(perms, dict):
        perms = {}
    allow = (
        [str(x) for x in perms["allow"]]
        if isinstance(perms.get("allow"), list)
        else []
    )
    if tool_name in allow:
        return True
    allow.append(tool_name)
    ask = (
        [str(x) for x in perms["ask"]]
        if isinstance(perms.get("ask"), list)
        else []
    )
    deny = (
        [str(x) for x in perms["deny"]]
        if isinstance(perms.get("deny"), list)
        else []
    )

    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("[permissions]"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("["):
                i += 1
            continue
        out.append(line)
        i += 1
    while out and not out[-1].strip():
        out.pop()
    section = "[permissions]\n"
    if allow:
        section += f"allow = {json.dumps(allow, ensure_ascii=False)}\n"
    if ask:
        section += f"ask = {json.dumps(ask, ensure_ascii=False)}\n"
    if deny:
        section += f"deny = {json.dumps(deny, ensure_ascii=False)}\n"
    body = "\n".join(out).rstrip() + "\n\n" + section
    try:
        path.write_text(body, encoding="utf-8")
    except OSError:
        return False
    return True
