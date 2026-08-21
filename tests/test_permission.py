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


# -- P2：权限规则表（config.toml [permissions]）+ 持久化批准 --------------------


def _named_tool(name: str, permission: str) -> ToolSchema:
    return ToolSchema(name, "desc", {"type": "object"}, permission)  # type: ignore[arg-type]


def test_rules_allow_overrides_confirm() -> None:
    from vgent.config import PermissionRules

    ps = PermissionSystem(rules=PermissionRules(allow=["shell"]))
    assert ps.check(_named_tool("shell", "exec"), {}) is Approval.AUTO


def test_rules_ask_forces_confirm_on_read() -> None:
    from vgent.config import PermissionRules

    ps = PermissionSystem(rules=PermissionRules(ask=["read_file"]))
    assert ps.check(_named_tool("read_file", "read"), {}) is Approval.NEED_CONFIRM


def test_rules_deny_wins_over_allow() -> None:
    from vgent.config import PermissionRules

    ps = PermissionSystem(rules=PermissionRules(allow=["shell"], deny=["shell"]))
    assert ps.check(_named_tool("shell", "exec"), {}) is Approval.DENIED


def test_rules_unmatched_falls_back_to_three_tier() -> None:
    from vgent.config import PermissionRules

    ps = PermissionSystem(rules=PermissionRules(ask=["other"]))
    assert ps.check(_tool("read"), {}) is Approval.AUTO
    assert ps.check(_tool("exec"), {}) is Approval.NEED_CONFIRM


def test_sticky_beats_ask_rule() -> None:
    from vgent.config import PermissionRules

    ps = PermissionSystem(rules=PermissionRules(ask=["shell"]))
    ps.approve_sticky("shell")
    assert ps.check(_named_tool("shell", "exec"), {}) is Approval.AUTO


def test_persist_allow_writes_and_preserves(tmp_path) -> None:
    import tomllib

    from vgent.permission import persist_allow

    p = tmp_path / "config.toml"
    p.write_text(
        '[provider]\nactive = "deepseek"\n\n[permissions]\nask = ["read_file"]\n',
        encoding="utf-8",
    )
    assert persist_allow(tmp_path, "shell") is True
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    assert data["permissions"]["allow"] == ["shell"]
    assert data["permissions"]["ask"] == ["read_file"]  # 同段其他键保留
    assert data["provider"]["active"] == "deepseek"  # 其他段保留
    # 幂等：已存在不重复
    assert persist_allow(tmp_path, "shell") is True
    assert tomllib.loads(p.read_text(encoding="utf-8"))["permissions"]["allow"] == ["shell"]


def test_persist_allow_missing_config_false(tmp_path) -> None:
    from vgent.permission import persist_allow

    assert persist_allow(tmp_path, "shell") is False
