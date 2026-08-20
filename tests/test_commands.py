"""M10 测试：外部命令加载（~/.vgent/commands/*.py 的 run(ctx, args)）。"""
from __future__ import annotations

from vgent.commands import load_commands


def _write(tmp_path, name: str, body: str):
    f = tmp_path / f"{name}.py"
    f.write_text(body, encoding="utf-8")
    return f


def test_loads_run_functions(tmp_path) -> None:
    _write(
        tmp_path,
        "hello",
        'def run(ctx, args: str) -> str:\n    return f"hi {args}"\n',
    )
    commands = load_commands(tmp_path)
    assert list(commands) == ["hello"]
    assert commands["hello"](None, "vgent") == "hi vgent"


def test_missing_dir_returns_empty(tmp_path) -> None:
    assert load_commands(tmp_path / "nope") == {}


def test_skips_broken_module(tmp_path) -> None:
    _write(tmp_path, "broken", "raise RuntimeError('boom')\n")
    assert load_commands(tmp_path) == {}


def test_skips_module_without_run(tmp_path) -> None:
    _write(tmp_path, "norun", "x = 1\n")
    assert load_commands(tmp_path) == {}


def test_skips_non_identifier_filename(tmp_path) -> None:
    _write(tmp_path, "bad name", "def run(ctx, args): return 'x'\n")
    assert load_commands(tmp_path) == {}


def test_skips_non_py_files(tmp_path) -> None:
    (tmp_path / "note.txt").write_text("不是命令", encoding="utf-8")
    assert load_commands(tmp_path) == {}
