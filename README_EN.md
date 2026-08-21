# vgent

[中文](README.md) | **English**

A general-purpose agent CLI (runtime harness) built from scratch by borrowing the best ideas from mature implementations — hermes-agent, openai-agents-python, OpenManus, MetaGPT, ag2, openclaw, gemini-cli and more. It is designed for daily local work: files, shell, and search, powered by an OpenAI-compatible model (DeepSeek by default, 1M context, multi-provider configurable).

**Status**: core features complete and daily-usable — REPL + local Web UI, five built-in tools, 3-tier permissions with persistent rules, 1M-token context management, task planning + agent state machine, failure reflection, cross-session memory (project-scoped), MCP client, headless one-shot execution.

## Features

- **Dual interface**: interactive REPL (prompt_toolkit + rich) **and** a local Web UI (`vgent --serve`, stdlib only, zero extra dependencies) sharing the same session storage.
- **Native tool calling**: JSON Schema tool definitions; built-in `shell` / `read_file` / `write_file` / `edit_file` / `search`.
- **3-tier permissions + rules**: read auto-approves, write/exec requires confirmation, unknown tiers are denied; confirm with `y` once / `a` always (per-session sticky) / `n` reject. `[permissions]` rules in `config.toml` persist across sessions (`allow` always, `ask` always confirm, `deny` rejects **and prunes the tool from what the model can see**); `/allow` writes approvals back to the config.
- **Context management**: within the 1M window, low-watermark free pruning (tool-result summaries, orphan tool-pair cleanup) + high-watermark compaction (TailWindow zero-cost / Summarize structured LLM summary, `tail` or `summarize` strategy, `/compact` manual trigger). SQLite keeps full history; compaction only affects the send list.
- **Task planning + state machine**: multi-step tasks get an auto-generated plan block, step statuses update and persist as execution proceeds (`/plan` to view, `/plan new` to redo).
- **Failure reflection loop**: after a tool failure the LLM produces an explicit reflection (Failure/Action) injected into the next round (`/reflect` manual trigger, persisted).
- **Cross-session memory**: task summaries stored as JSONL on this machine (`/remember` / `/recall` / `/memories`), auto-recall injection **scoped to the current project** (no cross-project leakage).
- **MCP client**: stdio connection to local MCP servers; tools appear with a `<server>_<tool>` prefix (`/mcp` to list).
- **Session persistence**: SQLite two-table + WAL; sessions can be listed / resumed / deleted; last session is remembered.
- **Project instructions**: user-level (`~/.vgent/AGENTS.md`) and project-level (`AGENTS.md`/`CLAUDE.md` found upward from the working directory) instructions are injected into the first LLM call; **user instructions come first**; user-defined slash commands via `~/.vgent/commands/<name>.py`.

## Quick start

1. **Install**:

   ```bash
   uv sync                 # in-project development
   # or global install:
   uv tool install .       # provides the `vgent` command (reinstall with --force after code updates)
   ```

2. **Configure your API key**: edit `~/.vgent/config.toml` (create if missing), minimal example:

   ```toml
   [provider]
   active = "deepseek"

   [providers.deepseek]
   base_url = "https://api.deepseek.com"
   model = "deepseek-v4-flash"
   api_key = "your key"        # or use api_key_env to point at an environment variable
   ```

   Without a key, conversations return 401 but the program never crashes.

3. **Launch**:

   ```bash
   uv run vgent            # CLI REPL (session picker → chat; /help for commands)
   uv run vgent --serve    # Web UI (opens browser at http://127.0.0.1:8477)
   uv run vgent --print "question"  # headless one-shot: run a turn and print the result
   ```

## Web UI

```bash
vgent --serve            # start the local web page (auto-opens browser, 127.0.0.1:8477)
vgent serve --port 8080  # `vgent serve` is an alias for --serve; --port sets the port
```

The browser page is a GUI REPL: session list on the left (new / resume / delete), chat area in the middle (streaming output, tool cards, collapsible thinking blocks), permission popups (y once / a always / n reject), and slash commands such as `/plan` `/compact` `/allow` `/remember`. Single-user localhost access, no auth; shares the same session storage as the CLI. Pure stdlib implementation, zero extra dependencies.

## CLI usage

Startup flags:

| Flag | Description |
|---|---|
| `--new` | Skip session selection, create a new session |
| `--resume [ID\|N\|last]` | Resume a session: default = last; N = list number; or a session id |
| `--list-sessions` | List sessions and exit |
| `--delete-session ID` | Delete a session and exit |
| `--provider <name>` | Temporarily switch provider (a `[providers]` name in config.toml) |
| `--serve [--port N]` | Start the local Web UI |
| `-p` / `--print <text>` | Headless: run one turn, print the result, exit (script/CI; write/exec default to reject) |
| `--version` | Version |

REPL commands:

| Command | Description |
|---|---|
| `/new` | New session |
| `/resume [last\|N\|id]` | List & resume; with an argument, switch directly (mirrors `--resume`) |
| `/list` | List sessions |
| `/delete` | Delete a session by number (current session cannot be deleted) |
| `/compact` | Compact the current session (LLM summary of the middle history, effective next turn) |
| `/plan` | View the task plan (`/plan new` clears and replans) |
| `/reflect` | Reflect on recent failures and generate corrective actions (LLM, persisted) |
| `/remember <topic>` | Remember the current session (LLM summary stored locally) |
| `/recall <keyword>` | Search memory and inject it into context |
| `/memories` | List remembered task summaries |
| `/mcp` | List loaded MCP tools |
| `/reasoning` | Toggle streaming display of model thinking (on/off) |
| `/allow <tool>` | Approve a tool (per-session sticky + persisted to config.toml) |
| `/help` `/exit` | Help / quit |

## Tools & permissions

| Tool | Tier | Description |
|---|---|---|
| `shell` | exec | Run shell commands (Git Bash on Windows; requires confirmation) |
| `write_file` | write | Write/append a file (auto-creates parent dirs; requires confirmation) |
| `edit_file` | write | Surgical edit: exact string match replace (unique match, replace_all, ambiguity/not-found errors fed back; requires confirmation) |
| `read_file` | read | Read a file (UTF-8, with line numbers; auto-approved) |
| `search` | read | Recursive regex search (skips .git/node_modules etc.; auto-approved) |

Confirmation flow: `y` run once / `a` always for this session (sticky) / `n` reject (the rejection is fed back to the model, which adjusts). Environments without an interactive confirm (pipes/headless) default to reject — a safe default.

Permission rules (`[permissions]` in `config.toml`): `allow` always runs without asking, `ask` always confirms (even read-tier tools), `deny` rejects **and prunes the tool from the model-visible tool set**; `/allow <tool>` persists approvals to config.toml across sessions. Rules that don't match fall back to the three tiers.

## Configuration

Full `~/.vgent/config.toml` example:

```toml
[provider]
active = "deepseek"            # currently active provider

[providers.deepseek]           # one section per provider
base_url = "https://api.deepseek.com"
model = "deepseek-v4-flash"    # DeepSeek V4 Flash, 1M context
api_key = ""                   # put the key here
api_key_env = "DEEPSEEK_API_KEY"  # or point at an env var (takes precedence over api_key; empty = file key only)

[context]                      # context engine
threshold_percent = 0.75       # high watermark: trigger compaction
prune_percent = 0.30           # low watermark: trigger free pruning
tail_token_budget = 20000      # tail token budget preserved during compaction
compact_strategy = "tail"      # tail (zero-cost) | summarize (LLM summary, needs /compact)

show_reasoning = false         # whether to stream model thinking by default (/reasoning toggles)
memory_auto = false            # auto-save a session summary when the task plan completes

[mcp.servers.echo]             # MCP server (stdio; tools registered with server_tool prefix)
command = "python"             # launch command
args = ["path/to/server.py"]
permission = "exec"            # default tier for this server's tools: read | write | exec

[permissions]                  # permission rules (/allow appends here)
allow = ["shell"]              # always run without asking
# ask = ["read_file"]          # always confirm (even read-tier)
# deny = ["write_file"]        # reject and hide from the model (schemas pruned)
```

Others: `log_level`, `data_dir` (default `~/.vgent` on this machine; override with the `VGENT_HOME` environment variable).

## Advanced features

- **Task planning**: multi-step tasks generate a plan on the first turn (`/plan` view, `/plan new` redo); step statuses update and persist with the session.
- **Context compaction**: `/compact` compresses the middle history into a structured summary (`<analysis>` draft + `<summary>` with required sections: unfinished tasks / key decisions / key facts / safety constraints preserved verbatim; falls back to thinking content when the body is a fragment, and to a TailWindow marker when too short). Auto-triggered at the high watermark (default 75%, configurable via `compact_strategy`).
- **Memory**: `/remember <topic>` stores, `/recall <keyword>` retrieves and injects, `/memories` lists; topic matches auto-inject a recall (not persisted), **scoped to the current project**; `memory_auto=true` saves a summary when the plan completes.
- **Reflection**: after a tool failure, one explicit reflection (Failure/Action) is injected to guide correction; `/reflect` triggers manually (persisted).
- **AGENTS.md instructions**: user-level (`~/.vgent/AGENTS.md`) and project-level (nearest `AGENTS.md`/`CLAUDE.md` found upward from the working directory, 8 levels / 8K cap) instructions are injected into the first LLM call — **user first, project second**; write your conventions into the file and they apply.
- **External commands**: `~/.vgent/commands/<name>.py` with `run(ctx, args: str) -> str` is callable as `/name args` in the REPL; built-ins take priority, broken files are skipped without blocking startup.

## Known limitations

- **Tool surface**: only `shell` / `read_file` / `write_file` / `edit_file` / `search`; no web fetch or browser control. MCP is a stdio-only client that rebuilds the connection per call (no persistent connection); streamable HTTP/SSE transport not implemented.
- **Context & memory**: memory retrieval is keyword substring matching (no tokenization or vector search); auto-summary is off by default (`memory_auto=false`); `/exit` does not auto-save (only explicit `/remember` or plan completion does).
- **Planning & reflection**: plan generation depends on model cooperation (best-effort; no plan if the model refuses); failure detection is a conservative heuristic that may miss or misfire.
- **Web UI**: single-user localhost without auth; per-session serial execution (input disabled while running); external commands are REPL-only; native styles replace rich rendering in the browser.
- **Platform**: under Git Bash/mintty or piped input, prompt_toolkit cannot access the Windows console, so the REPL falls back to plain `input()` (no multiline/history/completion; functionality unaffected). The Web UI is not affected.
- **Concurrency**: v1 is fully sync with serial per-session execution; no parallel tool calls yet. Async refactoring is deferred until there is a real need.
- **Explicitly out of scope**: multi-agent, vector semantic search, MCP server hosting, telemetry, prompt-YAML frameworks, Capability/Tool-Provider abstractions.

## Architecture

Eight components along the message flow (details in the code):

| # | Component | File | Responsibility |
|---|---|---|---|
| ① | Interface | `cli.py` / `web/` | REPL + Web UI (streaming render, command dispatch) |
| ② | Agent Loop | `agent.py` | Dialog state machine: chat → tool_calls → permission → execute → write back |
| ③ | LLM Provider | `llm.py` | openai SDK (sync streaming), tool_calls merge, usage reporting |
| ④ | Session storage | `store.py` | SQLite two-table + WAL, thread_id, agent state persistence |
| ⑤ | Context engine | `context.py` | usage counting, low-watermark pruning, high-watermark compaction (TailWindow/Summarize) |
| ⑥ | Tools | `tools.py` | JSON Schema + dispatcher, five built-in tools |
| ⑦ | Permissions | `permission.py` | 3 tiers + sticky confirm + rules table |
| ⑧ | Config | `config.py` | Loads `~/.vgent/config.toml` (providers / permission rules / MCP) |

v2 modules: `task.py` (task planning), `state.py` (state machine), `reflection.py` (reflection loop), `memory/episodic.py` (cross-session memory, project-scoped), `mcp/` (MCP client), `workspace.py` (AGENTS.md instructions), `commands.py` (external commands).

## Development

```bash
uv run pytest           # tests (311)
uv run ruff check .     # lint
```

- Tests inject FakeLLM/ScriptedLLM and never touch the network; real-model smoke tests are run separately.
- Each milestone is verified independently: FakeLLM unit tests + a real-model smoke test, then the build log is updated.

## License

[MIT](LICENSE)
