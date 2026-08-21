# vgent

**vgent** is a general-purpose agent CLI (runtime harness) built from scratch by borrowing the best ideas from mature implementations — hermes-agent, openai-agents-python, OpenManus, MetaGPT, ag2, openclaw, gemini-cli and more. It is designed for daily local work: files, shell, and search, powered by an OpenAI-compatible model (DeepSeek by default).

**Highlights**

- Dual interface: interactive REPL **and** a local Web UI (`vgent --serve`, zero extra dependencies)
- Native tool calling: `shell` / `read_file` / `write_file` / `search`
- 3-tier permissions (auto-approve / confirm / reject) with per-session sticky
- 1M-token context management: free pruning + threshold-triggered compaction (TailWindow / LLM summarization)
- Task planning + agent state machine, failure reflection loop, episodic cross-session memory
- MCP client, multi-provider config, SQLite session persistence (list / resume / delete)
- AGENTS.md project instructions + user-defined slash commands

**Known limitations**

- Single-user local Web UI (localhost, no auth); REPL falls back to plain `input()` under Git Bash/mintty
- Built-in tools are file/shell/search only (no web fetch / browser control); MCP client is stdio-only
- Memory retrieval is keyword substring matching (no vector search); task planning is best-effort

**Quick start**

```bash
uv sync                          # install dependencies
# put your API key in ~/.vgent/config.toml (see 配置 below)
uv run vgent                     # interactive REPL
uv run vgent --serve             # Web UI → http://127.0.0.1:8477
```

For the full documentation (Chinese), see below.

---

# vgent（中文版）

通用 agent CLI（运行时 harness）——参考 hermes-agent / openai-agents-python / OpenManus / MetaGPT / ag2 / openclaw / gemini-cli 等主流实现，各取所长，从零构建。默认模型 DeepSeek（OpenAI-compatible，1M 上下文），多 provider 可配置。

**状态**：核心功能完成并日常可用——REPL + Web UI 双界面、四内置工具、三档权限、1M 上下文管理、任务计划与状态机、失败反思、跨会话记忆、MCP 客户端。

## 特性

- **双界面**：交互式 REPL（prompt_toolkit + rich）与本地 Web UI（`vgent --serve`，stdlib 实现、零新增依赖）共用同一套会话存储。
- **原生 tool calling**：JSON Schema 定义工具，模型直接返回结构化 tool_calls；内置 `shell` / `read_file` / `write_file` / `search`。
- **三档权限**：读类自动放行、写/执行类确认、未知档拒绝；确认支持「y 一次 / a 本会话总是（sticky）/ n 拒绝」。
- **上下文管理**：1M 窗口内低水位免费剪枝（工具结果摘要、清孤儿 tool 对）+ 高水位触发压缩（TailWindow 零成本 / Summarize LLM 摘要，/compact 手动触发）；SQLite 保留全量历史，压缩只影响发送列表。
- **任务计划 + 状态机**：多步任务模型自动输出计划块，步骤状态随执行更新并持久化（/plan 查看）。
- **失败反思循环**：工具失败后 LLM 显式反思（Failure/Action）注入下一轮引导修正（/reflect 手动触发）。
- **跨会话记忆**：LLM 生成任务摘要存本机 JSONL（/remember /recall /memories），自动回忆注入。
- **MCP 客户端**：stdio 连本地 MCP server，工具以 `<server>_<tool>` 前缀进入工具面（/mcp 查看）。
- **会话持久化**：SQLite 双表 + WAL，会话可列出/恢复/删除，记忆上次会话。
- **AGENTS.md 项目指令**：启动时自动读取工作区指令注入首个调用；**外部命令**：`~/.vgent/commands/<name>.py` 定义 `/命令`。

## 快速开始

1. **安装**：

   ```bash
   uv sync                 # 项目内开发
   # 或全局安装：
   uv tool install .       # 生成全局命令 vgent（代码更新后 uv tool install . --force 重装）
   ```

2. **配置 API key**：编辑 `~/.vgent/config.toml`（不存在则新建），最小示例：

   ```toml
   [provider]
   active = "deepseek"

   [providers.deepseek]
   base_url = "https://api.deepseek.com"
   model = "deepseek-v4-flash"
   api_key = "你的 key"        # 或改用 api_key_env 指环境变量
   ```

   未配置 key 时对话会报 401，但程序不会崩溃。

3. **启动**：

   ```bash
   uv run vgent            # CLI REPL（会话选择 → 对话；/help 查看命令）
   uv run vgent --serve    # Web UI（自动打开浏览器，127.0.0.1:8477）
   ```

## Web UI

```bash
vgent --serve            # 启动本地 Web 页（自动打开浏览器，127.0.0.1:8477）
vgent serve --port 8080  # `vgent serve` 是 --serve 的别名写法；--port 指定端口
```

浏览器页面 = 带 GUI 的 REPL：左侧会话列表（新建/恢复/删除），中间对话区（流式输出、工具执行卡片、思考折叠块），工具权限弹窗（y 一次 / a 本会话总是 / n 拒绝），支持 `/plan` `/compact` `/remember` 等斜杠命令。单用户本地访问（127.0.0.1），无需鉴权；与 CLI 共用同一套会话存储。纯 stdlib 实现，零新增依赖。

## CLI 用法

启动参数：

| 参数 | 说明 |
|---|---|
| `--new` | 跳过会话选择，直接新建会话 |
| `--resume [ID\|N\|last]` | 恢复会话：缺省=上次；N=列表编号；或直接给会话 id |
| `--list-sessions` | 列出会话后退出 |
| `--delete-session ID` | 删除指定会话后退出 |
| `--provider <name>` | 临时切换 provider（config.toml 的 [providers] 名） |
| `--serve [--port N]` | 启动本地 Web UI |
| `--version` | 版本 |

REPL 命令：

| 命令 | 说明 |
|---|---|
| `/new` | 新建会话 |
| `/resume` `/list` | 列出并恢复会话 / 只列出 |
| `/delete` | 删除会话（按编号，当前会话不可删） |
| `/compact` | 压缩当前会话（LLM 摘要中间历史，下次对话生效） |
| `/plan` | 查看任务计划（`/plan new` 清除并重新规划） |
| `/reflect` | 反思最近失败，生成修正动作（LLM 分析，写入会话） |
| `/remember <主题>` | 记住当前会话（LLM 摘要存本机） |
| `/recall <关键词>` | 检索历史记忆并注入上下文 |
| `/memories` | 列出已记住的任务摘要 |
| `/mcp` | 列出已加载的 MCP 工具 |
| `/reasoning` | 切换思考过程展示（开/关） |
| `/help` `/exit` | 帮助 / 退出 |

## 工具与权限

| 工具 | 权限档 | 说明 |
|---|---|---|
| `shell` | exec | 执行 shell 命令（Windows 上为 Git Bash；需确认） |
| `write_file` | write | 写入/追加文件（自动建目录；需确认） |
| `read_file` | read | 读取文件（UTF-8，带行号；自动放行） |
| `search` | read | 递归正则搜索（自动跳过 .git/node_modules 等；自动放行） |

确认交互：`y` 执行一次 / `a` 本会话总是允许（sticky，之后不再问）/ `n` 拒绝（拒绝结果回喂模型，模型会调整方案）。无确认交互的环境（管道/headless）默认拒绝，保证安全。

## 配置

完整 `~/.vgent/config.toml` 示例：

```toml
[provider]
active = "deepseek"            # 当前激活的 provider

[providers.deepseek]           # 每个 provider 一段
base_url = "https://api.deepseek.com"
model = "deepseek-v4-flash"    # DeepSeek V4 Flash，1M 上下文
api_key = ""                   # 直接写 key
api_key_env = "DEEPSEEK_API_KEY"  # 或指环境变量（优先级高于 api_key；空 = 只用文件里的 key）

[context]                      # 上下文引擎（决策 8）
threshold_percent = 0.75       # 高水位：触发压缩
prune_percent = 0.30           # 低水位：触发免费剪枝
tail_token_budget = 20000      # 压缩时尾部保留 token 预算
compact_strategy = "tail"      # tail（零成本）| summarize（LLM 摘要，依赖 /compact）

show_reasoning = false         # 默认是否流式展示模型思考过程（/reasoning 可切换）
memory_auto = false            # 任务计划完成时自动存会话摘要（每会话一次）

[mcp.servers.echo]             # MCP 服务器（stdio 拉起；工具以 server_tool 前缀注册）
command = "python"             # 启动命令
args = ["path/to/server.py"]
permission = "exec"            # 该 server 工具默认权限档：read | write | exec
```

其他：`log_level`（日志级别）、`data_dir`（数据目录，默认本机 `~/.vgent`，可用环境变量 `VGENT_HOME` 覆盖）。

## 进阶功能

- **任务计划**：多步任务首轮自动生成计划（`/plan` 查看，`/plan new` 重做），步骤状态随执行更新并随会话持久化。
- **上下文压缩**：`/compact` 用 LLM 把中间历史压成摘要；高水位自动触发（默认 75%，`compact_strategy` 可配）。
- **记忆**：`/remember <主题>` 存、`/recall <关键词>` 检索注入、`/memories` 列出；命中主题时自动注入回忆（不落库）。
- **AGENTS.md 项目指令**：启动时从工作目录向上找最近的 `AGENTS.md`（`CLAUDE.md` 兜底，8 层上限、8K 截断），注入首个 LLM 调用——把项目约定写进去即可生效。
- **外部命令**：`~/.vgent/commands/<name>.py` 定义 `run(ctx, args: str) -> str`，REPL 里 `/name 参数` 调用；内置命令优先，坏文件跳过不阻塞启动。

## 已知限制与不足

- **工具面**：内置工具仅 `shell` / `read_file` / `write_file` / `search`；web fetch、浏览器控制未实现（v1 决策范围）。MCP 仅 stdio 客户端，每次调用重建连接（无常驻连接），streamable HTTP/SSE 传输未做。
- **上下文与记忆**：记忆检索是关键词子串匹配（无分词、无向量语义检索），需提到完整主题词才命中；自动摘要默认关闭（`memory_auto=false`）；`/exit` 不自动存记忆（显式 `/remember` 或任务计划完成才存）。
- **任务计划与反思**：计划生成依赖模型配合（best-effort，模型不配合则无计划）；反思失败判定为启发式，可能漏判/误判（保守偏向不触发）。
- **Web UI**：单用户 localhost 无鉴权，仅限本机使用；同一会话串行（转圈时禁输入）；外部命令（`~/.vgent/commands`）仅在 REPL 可用；浏览器侧以原生样式替代 rich 渲染。
- **平台**：Git Bash/mintty 或管道输入下 prompt_toolkit 拿不到 Windows 控制台，REPL 自动退回基础 `input()`（无多行编辑/历史/补全，功能不受影响）；Web UI 不受此影响。
- **并发**：v1 全 sync（决策 10），单会话内串行执行，尚无并行工具调用；async 重构留待有真实并行需求时再做。
- **明确不做（v2 范围外）**：多 Agent、向量语义检索、MCP server 宿主、遥测平台、prompt YAML 框架化、Capability/Tool Provider 抽象。

## 架构

八个组件沿消息流排列（详见代码注释）：

| # | 组件 | 文件 | 职责 |
|---|---|---|---|
| ① | 界面层 | `cli.py` / `web/` | REPL + Web UI（流式渲染、命令分发） |
| ② | Agent Loop | `agent.py` | 对话状态机：chat → tool_calls → 权限 → 执行 → 写回 |
| ③ | LLM Provider | `llm.py` | openai SDK（sync 流式）、tool_calls 合并、usage 上报 |
| ④ | 会话存储 | `store.py` | SQLite 双表 + WAL、thread_id、Agent 状态落库 |
| ⑤ | 上下文引擎 | `context.py` | usage 计数、低水位剪枝、高水位压缩（TailWindow/Summarize） |
| ⑥ | 工具层 | `tools.py` | JSON Schema + dispatcher、内置四工具 |
| ⑦ | 权限系统 | `permission.py` | 三档 + sticky 确认 |
| ⑧ | 配置 | `config.py` | 加载 `~/.vgent/config.toml`（多 provider） |

v2 演进模块：`task.py`（任务计划）、`state.py`（状态机）、`reflection.py`（反思循环）、`memory/episodic.py`（跨会话记忆）、`mcp/`（MCP 客户端）、`workspace.py`（AGENTS.md 指令）、`commands.py`（外部命令）。

## 开发

```bash
uv run pytest           # 测试（170 例）
uv run ruff check .     # lint
```

- 测试用 FakeLLM/ScriptedLLM 注入，不触网；真实模型冒烟另跑。
- 每个里程碑独立验证：FakeLLM 单测 + 真实模型冒烟，完成后回写构建日志。

## License

[MIT](LICENSE)
