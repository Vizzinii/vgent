# vgent

通用 agent CLI（运行时 harness）——参考 hermes-agent / openai-agents-python / OpenManus / MetaGPT 等主流实现，各取所长。默认模型 DeepSeek（OpenAI-compatible）。

**状态**：**v1 完成**——M0-M4 ✅ + M5 收尾 ✅（write_file/search 补齐、错误重试、测试补强，2026-08-20）。设计文档与跨机交接见 [HANDOFF.md](HANDOFF.md)。

内置工具：`shell`（exec 档，需确认）、`write_file`（write 档，需确认）、`read_file` / `search`（read 档，自动放行）。REPL 里 `/reasoning` 可切换流式展示模型思考过程（默认关；`config.toml` 顶层 `show_reasoning = true` 可默认开）。

## 运行

```bash
uv sync                 # 首次：安装依赖
uv run vgent            # 启动 REPL（会话选择 → 对话；/help 查看命令）
uv run pytest           # 测试
uv run ruff check .     # lint
```

## 安装为全局命令（可选）

```bash
uv tool install .           # 全局安装：~/.local/bin/vgent（代码更新后加 --force 重装）
vgent --list-sessions       # 非交互：列出会话
vgent --resume              # 恢复上次会话（--resume 2 = 列表编号；--resume <id>）
vgent --delete-session ID   # 非交互：删除会话
vgent --new                 # 跳过会话选择，直接新建
vgent --provider <name>     # 临时切换 provider
```

配置在 `~/.vgent/config.toml`：多 provider（`[providers.<name>]` 定义、`[provider] active` 选择，启动可 `vgent --provider <name>` 临时切换；api_key 用文件字段或每 provider 独立的 `api_key_env` 环境变量）。未配置 key 时对话会报 401，但 REPL 不崩溃。

## 双机开发注意事项（M1 新增）

- 本仓库在百度网盘同步盘：**`.venv` 不要进同步盘**。两台机器各自 `uv sync`，且先设置环境变量 `UV_PROJECT_ENVIRONMENT` 指向本机路径（如 `%USERPROFILE%\.vgent\venv-vgent`）。
- **`.python-version` 钉 3.12.13 是有意的**：项目路径含中文，Python ≤3.12.2 的 site.py 用 locale 编码（中文系统=GBK）读 editable 安装的 `.pth` 会崩溃（`PYTHONUTF8=1` 也无效）；3.12.13+ 改为 utf-8-sig 优先读，无此问题。另一台机器 `uv sync` 会自动下载 3.12.13。
- **Windows 输入适配**：Git Bash/mintty 或管道输入下 prompt_toolkit 拿不到 Windows 控制台（`NoConsoleScreenBufferError`），vgent 自动退回 `input()`；完整交互（多行编辑/历史）请在 Windows Terminal 或 cmd 里运行。
- 跨机交接：一切状态以 HANDOFF.md 为准（同步盘内）；每里程碑分对话执行，完成后回写「构建日志」。
- 百度网盘同步瞬间会锁文件，写入报 EBUSY 时等几秒重试。
