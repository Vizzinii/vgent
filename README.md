# vgent

**[中文]** | [English](README_EN.md)

通用 agent CLI（运行时 harness）——参考 hermes-agent / openai-agents-python / OpenManus / MetaGPT / ag2 / openclaw / gemini-cli 等主流实现，各取所长，从零构建。默认模型 DeepSeek（OpenAI-compatible，1M 上下文），多 provider 可配置。

**状态**：核心功能完成并日常可用——REPL + Web UI 双界面、五内置工具、三档权限 + 可持久化规则、1M 上下文管理、任务计划与状态机、失败反思、跨会话记忆（按项目隔离）、快照/恢复、MCP 客户端、headless 单次执行。

## 特性

- **双界面**：交互式 REPL（prompt_toolkit + rich）与本地 Web UI（`vgent --serve`，stdlib 实现、零新增依赖）共用同一套会话存储。
- **原生 tool calling**：JSON Schema 定义工具，模型直接返回结构化 tool_calls；内置 `shell` / `read_file` / `write_file` / `edit_file` / `search`。
- **三档权限 + 规则表**：读类自动放行、写/执行类确认、未知档拒绝；确认支持「y 一次 / a 本会话总是（sticky）/ n 拒绝」；`config.toml` 的 `[permissions]` 规则可持久化（allow 始终放行 / ask 总是确认 / deny 拒绝且从模型可见的工具池裁剪），`/allow` 把批准写回配置跨会话记住。
- **上下文管理**：1M 窗口内低水位免费剪枝（工具结果摘要、清孤儿 tool 对）+ 高水位触发压缩（TailWindow 零成本 / Summarize LLM 结构化摘要，`/compact` 手动触发）；发送前 tiktoken 精确估算（含 tools schema 固定开销 + 预留输出 token，离线自动回退启发式）；压缩结果持久化，恢复会话后仍以压缩底稿续聊（不发全量历史）；SQLite 保留全量历史。
- **任务计划 + 状态机**：多步任务模型自动输出计划块，步骤状态随执行更新并持久化（`/plan` 查看、`/plan new` 重做）。
- **失败反思循环**：工具失败后 LLM 显式反思（Failure/Action）注入下一轮引导修正（`/reflect` 手动触发）。
- **跨会话记忆**：LLM 生成任务摘要存本机 JSONL（`/remember` / `/recall` / `/memories`），自动回忆注入且**按项目隔离**（防跨项目串味）；`memory_auto=true` 时启用**自动两阶段记忆管线**——每轮后台抽取（密钥黑名单过滤）+ 防抖合并（≥3 信号或空闲 5 分钟）成 `MEMORY.md`（注册表）/ `memory_summary.md`（注入 system 的短总览），`/memory` 查看管理，`memory_read` / `memory_grep` 工具按需检索。
- **快照/恢复**：`write_file`/`edit_file` 写盘前自动登记原文（sha256 去重），每回合末封存为版本化快照（同一文件跨回合可回溯多个版本）；`/snapshot [名]` 拍命名档、`/restore last|编号|名|undo` 恢复（只撤文件、不动对话）；崩溃后残留回合自动提升为快照（claude fileHistory 思路）。
- **MCP 客户端**：stdio 连本地 MCP server，工具以 `<server>_<tool>` 前缀进入工具面（`/mcp` 查看）。
- **会话持久化**：SQLite 双表 + WAL，会话可列出/恢复/删除，记忆上次会话。
- **项目指令**：启动时自动读取用户级（`~/.vgent/AGENTS.md`）与项目级（工作区向上找 `AGENTS.md`/`CLAUDE.md`）指令注入首个调用；**外部命令**：`~/.vgent/commands/<name>.py` 定义 `/命令`。

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
   uv run vgent --print "问题"  # 无头跑一轮对话并输出结果（脚本/CI 用）
   ```

## Web UI

```bash
vgent --serve            # 启动本地 Web 页（自动打开浏览器，127.0.0.1:8477）
vgent serve --port 8080  # `vgent serve` 是 --serve 的别名写法；--port 指定端口
```

浏览器页面 = 带 GUI 的 REPL：左侧会话列表（新建/恢复/删除），中间对话区（流式输出、工具执行卡片、思考折叠块），工具权限弹窗（y 一次 / a 本会话总是 / n 拒绝），支持 `/plan` `/compact` `/allow` `/remember` 等斜杠命令。单用户本地访问（127.0.0.1），无需鉴权；与 CLI 共用同一套会话存储。纯 stdlib 实现，零新增依赖。

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
| `-p` / `--print <问题>` | 无头跑一轮对话并输出结果后退出（脚本/CI 用；write/exec 默认拒绝） |
| `--version` | 版本 |

REPL 命令：

| 命令 | 说明 |
|---|---|
| `/new` | 新建会话 |
| `/resume [last\|N\|id]` | 列出并恢复会话；带参直接切换（对齐 CLI `--resume`） |
| `/list` | 列出会话 |
| `/delete` | 删除会话（按编号，当前会话不可删） |
| `/compact` | 压缩当前会话（LLM 摘要中间历史，下次对话生效） |
| `/plan` | 查看任务计划（`/plan new` 清除并重新规划） |
| `/reflect` | 反思最近失败，生成修正动作（LLM 分析，写入会话） |
| `/remember <主题>` | 记住当前会话（LLM 摘要存本机） |
| `/recall <关键词>` | 检索历史记忆并注入上下文 |
| `/memories` | 列出已记住的任务摘要 |
| `/memory [子命令]` | 项目长期记忆：无参=总览+管线状态；`show`（summary 全文）/ `path` / `grep <词>` / `clear` |
| `/mcp` | 列出已加载的 MCP 工具 |
| `/reasoning` | 切换思考过程展示（开/关） |
| `/allow <工具>` | 放行工具（本会话 sticky + 写入 config.toml 跨会话记住） |
| `/snapshot [名]` | 把本会话改过的文件拍成命名档（无名用时间戳） |
| `/restore` | 列出可恢复的快照（`/restore last\|编号\|名\|undo` 恢复；只撤文件不动对话） |
| `/help` `/exit` | 帮助 / 退出 |

## 工具与权限

| 工具 | 权限档 | 说明 |
|---|---|---|
| `shell` | exec | 执行 shell 命令（Windows 上为 Git Bash；需确认） |
| `write_file` | write | 写入/追加文件（自动建目录；需确认） |
| `edit_file` | write | 手术式编辑：精确字符串匹配替换（唯一匹配、replace_all、多义/未找到报错回喂；需确认） |
| `read_file` | read | 读取文件（UTF-8，带行号；自动放行） |
| `search` | read | 递归正则搜索（自动跳过 .git/node_modules 等；自动放行） |
| `memory_read` | read | 读项目长期记忆文件（MEMORY.md / rollout_summaries/...；拒绝重读 memory_summary.md；自动放行） |
| `memory_grep` | read | 项目记忆关键词搜索（空格分隔 AND；自动放行） |

确认交互：`y` 执行一次 / `a` 本会话总是允许（sticky，之后不再问）/ `n` 拒绝（拒绝结果回喂模型，模型会调整方案）。无确认交互的环境（管道/headless）默认拒绝，保证安全。

权限规则表（`config.toml` 的 `[permissions]`）：`allow` 始终放行（不确认）、`ask` 总是确认（即使 read 档）、`deny` 拒绝且工具从模型可见的工具池中裁剪（模型根本看不到）；`/allow <工具>` 把批准写回 config.toml 跨会话记住。未命中规则回落三档。

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
light_model = ""               # 轻量模型（记忆抽取等后台任务）；空 = 用 model

[context]                      # 上下文引擎
threshold_percent = 0.75       # 高水位：触发压缩
prune_percent = 0.30           # 低水位：触发免费剪枝
tail_token_budget = 20000      # 压缩时尾部保留 token 预算
compact_strategy = "tail"      # tail（零成本）| summarize（LLM 摘要，依赖 /compact）
reserved_output_tokens = 0     # 发送前预留的模型输出 token；0 = 按窗口 5% 自动（1M→50k）

show_reasoning = false         # 默认是否流式展示模型思考过程（/reasoning 可切换）
memory_auto = false            # 启用自动两阶段记忆管线（后台抽取 + 防抖合并；默认关）

[mcp.servers.echo]             # MCP 服务器（stdio 拉起；工具以 server_tool 前缀注册）
command = "python"             # 启动命令
args = ["path/to/server.py"]
permission = "exec"            # 该 server 工具默认权限档：read | write | exec

[permissions]                  # 权限规则（/allow 可追加写入）
allow = ["shell"]              # 始终放行（不确认）
# ask = ["read_file"]          # 总是确认（即使 read 档）
# deny = ["write_file"]        # 拒绝且模型看不到（schemas 裁剪）
```

其他：`log_level`（日志级别）、`data_dir`（数据目录，默认本机 `~/.vgent`，可用环境变量 `VGENT_HOME` 覆盖）。

## 进阶功能

- **任务计划**：多步任务首轮自动生成计划（`/plan` 查看，`/plan new` 重做），步骤状态随执行更新并随会话持久化。
- **上下文压缩**：`/compact` 用 LLM 把中间历史压成结构化摘要（`<analysis>` 草稿 + `<summary>` 必含小节：未完成任务/关键决策/关键事实/安全约束原样保留；正文碎片时自动回退思考流，过短判失败退回 TailWindow）；高水位自动触发（默认 75%，`compact_strategy` 可配）。
- **记忆**：`/remember <主题>` 存、`/recall <关键词>` 检索注入、`/memories` 列出；命中主题时自动注入回忆（不落库，>1 天的旧条目附"请对照当前代码验证"警告），**按当前项目隔离**（防跨项目串味）；`memory_auto=true` 时启用**自动两阶段记忆管线**（每轮后台抽取 → 防抖合并 MEMORY.md + memory_summary.md，密钥黑名单过滤，退出时 drain 排空），`/memory` 命令族查看/搜索/清空，`memory_read` / `memory_grep` 工具让模型按需检索。
- **反思循环**：工具失败后自动注入一次显式反思（Failure/Action）引导修正；`/reflect` 手动触发（结果落库）。
- **AGENTS.md 指令**：用户级（`~/.vgent/AGENTS.md`）与项目级（从工作目录向上找最近的 `AGENTS.md`/`CLAUDE.md`，8 层上限、8K 截断）指令注入首个 LLM 调用——**用户指令在前、项目指令在后**，把约定写进文件即可生效。
- **外部命令**：`~/.vgent/commands/<name>.py` 定义 `run(ctx, args: str) -> str`，REPL 里 `/name 参数` 调用；内置命令优先，坏文件跳过不阻塞启动。

## 已知限制与不足

- **工具面**：内置工具仅 `shell` / `read_file` / `write_file` / `edit_file` / `search`；web fetch、浏览器控制未实现。MCP 仅 stdio 客户端，每次调用重建连接（无常驻连接），streamable HTTP/SSE 传输未做。
- **上下文与记忆**：记忆检索是关键词子串匹配（无分词、无向量语义检索），需提到完整主题词才命中；自动记忆管线默认关闭（`memory_auto=false`；`/remember` 手动照常）；`/exit` 不自动存记忆（显式 `/remember` 或管线抽取才存）。
- **任务计划与反思**：计划生成依赖模型配合（best-effort，模型不配合则无计划）；反思失败判定为启发式，可能漏判/误判（保守偏向不触发）。
- **快照/恢复**：只跟踪工作区内（相对启动目录）`write_file`/`edit_file` 改过的文件，bash 改文件不在快照内；每会话保留最近 20 个回合快照 + 20 个命名档（超限淘汰最旧），blob 引用计数 GC，超期 30 天未活动的会话快照目录自动清理；`/restore` 在 Web 端无确认交互直接执行（CLI 有确认）。
- **Web UI**：单用户 localhost 无鉴权，仅限本机使用；同一会话串行（转圈时禁输入）；外部命令仅在 REPL 可用；浏览器侧以原生样式替代 rich 渲染。
- **平台**：Git Bash/mintty 或管道输入下 prompt_toolkit 拿不到 Windows 控制台，REPL 自动退回基础 `input()`（无多行编辑/历史/补全，功能不受影响）；Web UI 不受此影响。
- **并发**：v1 全 sync，单会话内串行执行，尚无并行工具调用；async 重构留待有真实并行需求时再做。
- **明确不做**：多 Agent、向量语义检索、MCP server 宿主、遥测平台、prompt YAML 框架化、Capability/Tool Provider 抽象。

## 架构

八个组件沿消息流排列（详见代码注释）：

| # | 组件 | 文件 | 职责 |
|---|---|---|---|
| ① | 界面层 | `cli.py` / `web/` | REPL + Web UI（流式渲染、命令分发） |
| ② | Agent Loop | `agent.py` | 对话状态机：chat → tool_calls → 权限 → 执行 → 写回 |
| ③ | LLM Provider | `llm.py` | openai SDK（sync 流式）、tool_calls 合并、usage 上报 |
| ④ | 会话存储 | `store.py` | SQLite 双表 + WAL、thread_id、Agent 状态落库 |
| ⑤ | 上下文引擎 | `context.py` | usage 计数、低水位剪枝、高水位压缩（TailWindow/Summarize） |
| ⑥ | 工具层 | `tools.py` | JSON Schema + dispatcher、内置五工具 |
| ⑦ | 权限系统 | `permission.py` | 三档 + sticky 确认 + 规则表 |
| ⑧ | 配置 | `config.py` | 加载 `~/.vgent/config.toml`（多 provider / 权限规则 / MCP） |

v2 演进模块：`task.py`（任务计划）、`state.py`（状态机）、`reflection.py`（反思循环）、`memory/`（跨会话记忆：`episodic.py` 条目 + `store.py`/`prompts.py`/`pipeline.py`/`tools.py` 自动两阶段管线）、`snapshot.py`（快照/恢复）、`mcp/`（MCP 客户端）、`workspace.py`（AGENTS.md 指令）、`commands.py`（外部命令）。

## 开发

```bash
uv run pytest           # 测试（415 例）
uv run ruff check .     # lint
```

- 测试用 FakeLLM/ScriptedLLM 注入，不触网；真实模型冒烟另跑。
- 每个里程碑独立验证：FakeLLM 单测 + 真实模型冒烟，完成后回写构建日志。

## License

[MIT](LICENSE)
