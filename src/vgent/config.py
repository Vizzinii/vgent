"""⑧ 配置：从本机 ~/.vgent/config.toml 加载（决策 7：配置不进同步盘）。

多 provider（决策 2）：`[providers.<name>]` 定义、`[provider] active` 选择，
未写 active 时缺省用 deepseek（或第一个定义的 provider）；CLI 可 `--provider <name>` 覆盖。
兼容旧式单 provider 写法：字段直接写在 `[provider]` 下，等价 name=default。

api_key 优先级：provider 的 api_key_env 指定的环境变量 > 该 provider 的 api_key 字段
（api_key_env 缺省为空 = 只用文件里的 api_key，避免默认环境变量串到别的 provider）。
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path(os.environ.get("VGENT_HOME") or Path.home() / ".vgent")
DEFAULT_CONFIG_PATH = DEFAULT_DATA_DIR / "config.toml"

DEFAULT_PROVIDER = "deepseek"

_PROVIDER_FIELDS = ("base_url", "api_key", "api_key_env", "model", "context_length")


@dataclass
class ProviderConfig:
    name: str = DEFAULT_PROVIDER
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    api_key_env: str = ""  # 该 provider 的环境变量名（优先级高于 api_key）；空 = 不用环境变量
    model: str = "deepseek-v4-flash"
    context_length: int = 1_000_000  # DeepSeek V4 Flash 1M（用户确认）

    def api_key_resolved(self) -> str:
        """环境变量优先，其次 config.toml。"""
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or self.api_key
        return self.api_key


@dataclass
class ContextConfig:
    threshold_percent: float = 0.75  # 高水位：触发 LLM 摘要压缩（决策 8）
    prune_percent: float = 0.30  # 低水位：触发免费剪枝（1M 窗口主角）
    tail_token_budget: int = 20_000  # 压缩时尾部保留预算（hermes 默认 ~20K）
    compact_strategy: str = "tail"  # "tail" 零成本 | "summarize" LLM 摘要（M4；需注入 summarizer）


@dataclass
class Config:
    data_dir: Path = DEFAULT_DATA_DIR
    log_level: str = "INFO"
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    show_reasoning: bool = False  # M5：流式展示模型思考过程（/reasoning 可随时切换）

    def api_key_resolved(self) -> str:
        return self.provider.api_key_resolved()


def load_config(path: Path | None = None, provider: str | None = None) -> Config:
    """读取 config.toml 覆盖默认值；文件不存在则返回默认配置。

    provider 参数（CLI --provider）优先于文件里的 active。
    """
    cfg = Config()
    p = path or DEFAULT_CONFIG_PATH
    if not p.exists():
        if provider is not None and provider != DEFAULT_PROVIDER:
            raise ValueError(f"没有 config.toml，无法选择 provider '{provider}'")
        return cfg
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    if "data_dir" in data:
        cfg.data_dir = Path(data["data_dir"])
    if "log_level" in data:
        cfg.log_level = data["log_level"]
    if "show_reasoning" in data:
        cfg.show_reasoning = bool(data["show_reasoning"])
    if "context" in data:
        for key, value in data["context"].items():
            setattr(cfg.context, key, value)

    providers = _resolve_providers(data)
    if providers:
        active = provider or data.get("provider", {}).get("active")
        if active is None:
            active = (
                DEFAULT_PROVIDER
                if DEFAULT_PROVIDER in providers
                else next(iter(providers))
            )
        if active not in providers:
            raise ValueError(f"config.toml 激活的 provider '{active}' 未在 [providers] 中定义")
        for key, value in providers[active].items():
            if key in _PROVIDER_FIELDS:
                setattr(cfg.provider, key, value)
        cfg.provider.name = active
    return cfg


def _resolve_providers(data: dict) -> dict[str, dict]:
    """[providers.<name>] 定义 + 旧式 [provider] 字段写法（等价 name=default）。"""
    providers: dict[str, dict] = {
        name: dict(section) for name, section in data.get("providers", {}).items()
    }
    psec = data.get("provider", {})
    if any(k in psec for k in _PROVIDER_FIELDS):
        providers.setdefault("default", {}).update(
            {k: v for k, v in psec.items() if k in _PROVIDER_FIELDS}
        )
    return providers
