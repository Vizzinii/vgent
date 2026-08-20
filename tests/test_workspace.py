"""M10 测试：工作区指令（AGENTS.md / CLAUDE.md）发现。"""
from __future__ import annotations

from vgent.workspace import find_instructions


def test_find_agents_in_cwd(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("这是项目指令", encoding="utf-8")
    name, content = find_instructions(tmp_path)
    assert name == "AGENTS.md"
    assert content == "这是项目指令"


def test_find_claude_as_fallback(tmp_path) -> None:
    (tmp_path / "CLAUDE.md").write_text("claude 指令", encoding="utf-8")
    name, content = find_instructions(tmp_path)
    assert name == "CLAUDE.md"
    assert content == "claude 指令"


def test_agents_priority_over_claude(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("agents 指令", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("claude 指令", encoding="utf-8")
    name, _ = find_instructions(tmp_path)
    assert name == "AGENTS.md"


def test_walks_up_parents(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("上级指令", encoding="utf-8")
    child = tmp_path / "a" / "b"
    child.mkdir(parents=True)
    name, content = find_instructions(child)
    assert name == "AGENTS.md"
    assert content == "上级指令"


def test_missing_returns_none(tmp_path) -> None:
    assert find_instructions(tmp_path) is None


def test_empty_file_skipped(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("   \n", encoding="utf-8")
    assert find_instructions(tmp_path) is None


def test_oversize_truncated(tmp_path) -> None:
    (tmp_path / "AGENTS.md").write_text("x" * 20_000, encoding="utf-8")
    _, content = find_instructions(tmp_path)
    assert content.startswith("x" * 8_000)  # 正文截断到上限
    assert "已截断" in content
    assert len(content) < 20_000
