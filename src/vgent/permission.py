"""⑦ 权限/确认系统 —— v1 三档（决策 7，用户拍板）。

- read：自动放行
- write / exec：需确认；确认时可 sticky 放行（本会话内该工具不再问）
契约③ 的 `confirm(tool, args) -> bool` 在 M2 细化为三态 ConfirmResult
（APPROVE 一次 / ALWAYS 本会话 sticky / REJECT），满足「执行类确认+sticky」UX。
确认交互由 CLI 注入（rich prompt）；未注入时默认拒绝（headless 安全默认）。
"""
from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from vgent.tools import ToolSchema


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
    ) -> None:
        self._confirm = confirm
        self._sticky: set[str] = set()

    def check(self, tool: ToolSchema, args: dict) -> Approval:
        """工具执行前的检查：read 自动放行；write/exec 需确认；sticky 后自动放行。"""
        if tool.permission == "read":
            return Approval.AUTO
        if tool.name in self._sticky:
            return Approval.AUTO
        if tool.permission in ("write", "exec"):
            return Approval.NEED_CONFIRM
        return Approval.DENIED  # 未知档位：默认拒绝

    def confirm(self, tool: ToolSchema, args: dict) -> ConfirmResult:
        """走到确认交互；ALWAYS 时自动 sticky（本会话内）。无交互则拒绝。"""
        if tool.name in self._sticky:
            return ConfirmResult.APPROVE
        if self._confirm is None:
            return ConfirmResult.REJECT
        result = self._confirm(tool, args)
        if result is ConfirmResult.ALWAYS:
            self._sticky.add(tool.name)
        return result

    def approve_sticky(self, tool_name: str) -> None:
        """外部直接 sticky（供测试/未来 deny 列表/配置化使用）。"""
        self._sticky.add(tool_name)
