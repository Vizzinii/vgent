"""M2 测试：权限三档 + sticky。"""
from __future__ import annotations

from vgent.permission import Approval, ConfirmResult, PermissionSystem
from vgent.tools import ToolSchema


def _tool(permission: str) -> ToolSchema:
    return ToolSchema(f"t_{permission}", "desc", {"type": "object"}, permission)  # type: ignore[arg-type]


def test_read_auto_allow() -> None:
    ps = PermissionSystem()
    assert ps.check(_tool("read"), {}) is Approval.AUTO


def test_write_and_exec_need_confirm() -> None:
    ps = PermissionSystem()
    assert ps.check(_tool("write"), {}) is Approval.NEED_CONFIRM
    assert ps.check(_tool("exec"), {}) is Approval.NEED_CONFIRM


def test_no_confirm_rejects() -> None:
    ps = PermissionSystem()
    assert ps.confirm(_tool("exec"), {}) is ConfirmResult.REJECT


def test_sticky_auto_after_approve() -> None:
    ps = PermissionSystem()
    t = _tool("exec")
    ps.approve_sticky(t.name)
    assert ps.check(t, {}) is Approval.AUTO
    # sticky 后 confirm 不再走交互，直接 APPROVE
    assert ps.confirm(t, {}) is ConfirmResult.APPROVE


def test_confirm_callback_always_sticks() -> None:
    seen: list[str] = []

    def cb(tool, args):
        seen.append(tool.name)
        return ConfirmResult.ALWAYS

    ps = PermissionSystem(confirm=cb)
    t = _tool("exec")
    assert ps.confirm(t, {}) is ConfirmResult.ALWAYS
    assert ps.check(t, {}) is Approval.AUTO  # ALWAYS 已自动 sticky
    assert seen == [t.name]


def test_confirm_callback_reject_once() -> None:
    ps = PermissionSystem(confirm=lambda tool, args: ConfirmResult.REJECT)
    t = _tool("exec")
    assert ps.confirm(t, {}) is ConfirmResult.REJECT
    assert ps.check(t, {}) is Approval.NEED_CONFIRM  # 拒绝不 sticky，下次仍要问


def test_unknown_permission_denied() -> None:
    ps = PermissionSystem()
    t = ToolSchema("weird", "d", {"type": "object"}, "sudo")  # type: ignore[arg-type]
    assert ps.check(t, {}) is Approval.DENIED
