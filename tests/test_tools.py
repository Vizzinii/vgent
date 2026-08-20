"""M2 测试：工具注册/分发 + 内置 shell / read_file。"""
from __future__ import annotations

from vgent import tools
from vgent.tools import ToolRegistry, ToolSchema, default_tools


def test_registry_register_and_schemas() -> None:
    reg = ToolRegistry()
    reg.register(
        ToolSchema("echo_tool", "回显", {"type": "object"}, "read"),
        lambda a: "hi",
    )
    assert reg.get("echo_tool") is not None
    assert reg.get("nope") is None
    s = reg.schemas()[0]
    assert s["type"] == "function"
    assert s["function"]["name"] == "echo_tool"
    assert reg.execute("echo_tool", {}) == "hi"


def test_default_tools_registered() -> None:
    reg = default_tools()
    assert {s["function"]["name"] for s in reg.schemas()} == {
        "shell",
        "read_file",
        "write_file",
        "search",
    }


def test_shell_echo() -> None:
    reg = default_tools()
    out = reg.execute("shell", {"command": "echo vgent-ok"})
    assert out.startswith("exit 0")
    assert "vgent-ok" in out


def test_shell_missing_command() -> None:
    reg = default_tools()
    assert "缺少 command" in reg.execute("shell", {})


def test_shell_timeout() -> None:
    reg = default_tools()
    out = reg.execute("shell", {"command": "sleep 3", "timeout": 1})
    assert "超时" in out


def test_shell_failure_exit_code() -> None:
    reg = default_tools()
    out = reg.execute("shell", {"command": "exit 3"})
    assert out.startswith("exit 3")


def test_read_file_with_line_numbers(tmp_path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("l1\nl2\nl3\n", encoding="utf-8")
    reg = default_tools()
    out = reg.execute("read_file", {"path": str(p)})
    assert "1\tl1" in out
    assert "3\tl3" in out
    out = reg.execute("read_file", {"path": str(p), "offset": 2, "limit": 1})
    assert "2\tl2" in out
    assert "l1" not in out


def test_read_file_missing() -> None:
    reg = default_tools()
    out = reg.execute("read_file", {"path": "C:/no/such/file-vgent.txt"})
    assert "读取失败" in out


# -- M5：write_file / search ---------------------------------------------------


def test_write_file_overwrite_append_and_mkdir(tmp_path) -> None:
    reg = default_tools()
    p = tmp_path / "sub" / "a.txt"
    out = reg.execute("write_file", {"path": str(p), "content": "hello"})
    assert "已overwrite" in out
    assert p.read_text(encoding="utf-8") == "hello"
    # append
    reg.execute("write_file", {"path": str(p), "content": " world", "mode": "append"})
    assert p.read_text(encoding="utf-8") == "hello world"
    # overwrite 覆盖
    reg.execute("write_file", {"path": str(p), "content": "new"})
    assert p.read_text(encoding="utf-8") == "new"


def test_write_file_bad_args() -> None:
    reg = default_tools()
    assert "缺少 path" in reg.execute("write_file", {"content": "x"})
    out = reg.execute("write_file", {"path": "x.txt", "content": "y", "mode": "bad"})
    assert "mode" in out


def test_search_hits_and_misses(tmp_path) -> None:
    reg = default_tools()
    (tmp_path / "a.txt").write_text("hello vgent\nworld\nvgent again\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("nothing here\n", encoding="utf-8")
    out = reg.execute("search", {"pattern": "vgent", "path": str(tmp_path)})
    assert "a.txt:1: hello vgent" in out
    assert "a.txt:3: vgent again" in out
    assert "b.txt" not in out
    out2 = reg.execute("search", {"pattern": "zzz", "path": str(tmp_path)})
    assert "未找到" in out2


def test_search_regex_and_bad_pattern(tmp_path) -> None:
    reg = default_tools()
    (tmp_path / "c.txt").write_text("abc123\n", encoding="utf-8")
    out = reg.execute("search", {"pattern": r"\d+", "path": str(tmp_path)})
    assert "abc123" in out
    assert "正则错误" in reg.execute("search", {"pattern": "(", "path": str(tmp_path)})


def test_search_skips_noise_dirs(tmp_path) -> None:
    reg = default_tools()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.txt").write_text("secret vgent\n", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("public vgent\n", encoding="utf-8")
    out = reg.execute("search", {"pattern": "vgent", "path": str(tmp_path)})
    assert "keep.txt" in out
    assert ".git" not in out


def test_search_single_file(tmp_path) -> None:
    reg = default_tools()
    p = tmp_path / "only.txt"
    p.write_text("target line\n", encoding="utf-8")
    out = reg.execute("search", {"pattern": "target", "path": str(p)})
    assert "only.txt:1: target line" in out


# -- shell 解析兜底（真机首跑修复：Git 装在非标准路径） -------------------------


def test_git_roots_from_git_path(monkeypatch) -> None:
    """git 在 mingw64/bin 下（非 cmd）：向上推断到 Git 根目录。"""
    monkeypatch.setattr(
        "vgent.tools.shutil.which",
        lambda n: r"D:\git\Git\mingw64\bin\git.EXE" if n == "git" else None,
    )
    roots = tools._git_roots_from_git_path()
    assert any(r.name.lower() == "git" for r in roots)


def test_resolve_shell_via_registry(tmp_path, monkeypatch) -> None:
    """Git 装在非标准路径且 PATH 无 bash：靠注册表 InstallPath 找到 bash。"""
    fake = tmp_path / "Git" / "usr" / "bin" / "bash.exe"
    fake.parent.mkdir(parents=True)
    fake.write_text("fake bash")

    class _Key:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Winreg:
        HKEY_LOCAL_MACHINE = 1
        HKEY_CURRENT_USER = 2

        @staticmethod
        def OpenKey(hive, key):
            return _Key()

        @staticmethod
        def QueryValueEx(key, name):
            return (str(tmp_path / "Git"), 1)

    monkeypatch.setattr(tools, "winreg", _Winreg)
    monkeypatch.setattr(tools, "_SHELL_CANDIDATES", ())
    monkeypatch.setattr("vgent.tools.shutil.which", lambda n: None)  # PATH 里无 bash/git
    assert tools._resolve_shell() == str(fake)
