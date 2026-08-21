"""M0 冒烟 + 多 provider 配置测试。"""
import pytest

from vgent.config import Config, load_config


def test_default_config_values() -> None:
    cfg = Config()
    assert cfg.provider.name == "deepseek"
    assert cfg.provider.base_url == "https://api.deepseek.com"
    assert cfg.provider.api_key_env == ""
    assert cfg.provider.context_length == 1_000_000
    assert cfg.context.threshold_percent == 0.75
    assert cfg.context.prune_percent == 0.30
    assert cfg.context.tail_token_budget == 20_000


def test_load_missing_file_returns_defaults(tmp_path) -> None:
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.provider.model == "deepseek-v4-flash"


def test_load_toml_overrides(tmp_path) -> None:
    """旧式单 provider 写法（字段直接在 [provider] 下）仍可用。"""
    p = tmp_path / "config.toml"
    p.write_text(
        '[provider]\nmodel = "deepseek-chat"\n\n[context]\nthreshold_percent = 0.5\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.provider.model == "deepseek-chat"
    assert cfg.context.threshold_percent == 0.5


def test_api_key_env_precedence(tmp_path, monkeypatch) -> None:
    p = tmp_path / "config.toml"
    p.write_text(
        '[provider]\napi_key = "from-file"\napi_key_env = "DEEPSEEK_API_KEY"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert load_config(p).api_key_resolved() == "from-file"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    assert load_config(p).api_key_resolved() == "from-env"


def test_api_key_env_empty_uses_file_only(tmp_path, monkeypatch) -> None:
    """api_key_env 缺省为空：环境变量不串到未显式配置的 provider。"""
    p = tmp_path / "config.toml"
    p.write_text('[provider]\napi_key = "from-file"\n', encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    assert load_config(p).api_key_resolved() == "from-file"


def test_multi_provider_active(tmp_path) -> None:
    p = tmp_path / "config.toml"
    p.write_text(
        '[provider]\nactive = "tokenrhythm"\n\n'
        "[providers.deepseek]\nmodel = \"deepseek-v4-flash\"\n\n"
        "[providers.tokenrhythm]\n"
        'base_url = "https://tokenrhythm.studio/v1"\n'
        'model = "deepseek-v4-flash-0731"\n'
        'api_key = "sk-test"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.provider.name == "tokenrhythm"
    assert cfg.provider.base_url == "https://tokenrhythm.studio/v1"
    assert cfg.provider.model == "deepseek-v4-flash-0731"
    assert cfg.provider.api_key == "sk-test"


def test_multi_provider_cli_override(tmp_path) -> None:
    p = tmp_path / "config.toml"
    p.write_text(
        '[provider]\nactive = "tokenrhythm"\n\n'
        "[providers.deepseek]\nmodel = \"deepseek-v4-flash\"\n\n"
        "[providers.tokenrhythm]\nmodel = \"deepseek-v4-flash-0731\"\n",
        encoding="utf-8",
    )
    # --provider 优先于文件里的 active
    cfg = load_config(p, provider="deepseek")
    assert cfg.provider.name == "deepseek"
    assert cfg.provider.model == "deepseek-v4-flash"


def test_provider_specific_env_var(tmp_path, monkeypatch) -> None:
    p = tmp_path / "config.toml"
    p.write_text(
        '[provider]\nactive = "rhythm"\n\n'
        "[providers.rhythm]\n"
        'api_key = "from-file"\n'
        'api_key_env = "TOKENRHYTHM_API_KEY"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("TOKENRHYTHM_API_KEY", raising=False)
    assert load_config(p).api_key_resolved() == "from-file"

    monkeypatch.setenv("TOKENRHYTHM_API_KEY", "from-env")
    assert load_config(p).api_key_resolved() == "from-env"


def test_unknown_active_provider_raises(tmp_path) -> None:
    p = tmp_path / "config.toml"
    p.write_text('[provider]\nactive = "nope"\n\n[providers.deepseek]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="nope"):
        load_config(p)


def test_cli_provider_without_config_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="没有 config.toml"):
        load_config(tmp_path / "nonexistent.toml", provider="tokenrhythm")


def test_show_reasoning_config(tmp_path) -> None:
    assert Config().show_reasoning is False
    p = tmp_path / "config.toml"
    p.write_text("show_reasoning = true\n", encoding="utf-8")
    assert load_config(p).show_reasoning is True


def test_mcp_servers_parsed(tmp_path) -> None:
    """M9：配置解析 [mcp.servers.<name>]（command/args/permission；无 command 跳过）。"""
    p = tmp_path / "config.toml"
    p.write_text(
        '[mcp.servers.echo]\ncommand = "python"\nargs = ["s.py", "--x"]\n'
        '[mcp.servers.notes]\ncommand = "node"\npermission = "write"\n'
        '[mcp.servers.bad]\npermission = "read"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert set(cfg.mcp_servers) == {"echo", "notes"}
    echo = cfg.mcp_servers["echo"]
    assert echo.command == "python"
    assert echo.args == ["s.py", "--x"]
    assert echo.permission == "exec"  # 默认
    assert cfg.mcp_servers["notes"].permission == "write"


def test_mcp_bad_permission_defaults_exec(tmp_path) -> None:
    p = tmp_path / "config.toml"
    p.write_text('[mcp.servers.x]\ncommand = "python"\npermission = "sudo"\n', encoding="utf-8")
    assert load_config(p).mcp_servers["x"].permission == "exec"


def test_permissions_rules_parsed(tmp_path) -> None:
    """P2：[permissions] allow/ask/deny 数组解析。"""
    p = tmp_path / "config.toml"
    p.write_text(
        '[permissions]\nallow = ["shell"]\nask = ["read_file"]\ndeny = ["write_file"]\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.permissions.allow == ["shell"]
    assert cfg.permissions.ask == ["read_file"]
    assert cfg.permissions.deny == ["write_file"]


def test_permissions_invalid_or_missing(tmp_path) -> None:
    """P2：非数组或缺失的 [permissions] → 空规则（不影响默认三档）。"""
    p = tmp_path / "config.toml"
    p.write_text('[permissions]\nallow = "not-a-list"\n', encoding="utf-8")
    assert load_config(p).permissions.allow == []
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg.permissions.allow == [] and cfg.permissions.deny == []
