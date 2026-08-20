# vgent — 交接与进度记录（HANDOFF）

> 多端轮流工作用的状态文件。**在另一台机器上开工前，先读本文件**，确认上一位「我」问了什么、定了什么、停在哪。
> 工作区在百度网盘同步盘，本文件随盘同步。
> 最近更新：2026-08-20（**v1 收官 + 真机首跑修复**：M0-M5 全部完成（81 测试全绿 + tokenrhythm 冒烟）；真终端首跑验证三体验正常，修复 shell 非标准路径解析 / httpx2 日志噪音 / banner 富文本吞名 3 个 bug（83 passed）。下一步：v2 backlog——web UI / async 重构）

## 一句话定位

**vgent = 参考成熟实现、可日常使用的通用 agent CLI（运行时 harness，非评测框架），从零构建。**

## 当前阶段

**M0-M5 已 ✅ 全部完成（v1 收官）**。设计/框架阶段完成（决策 1-10 + 契约 v0.1 + MVP 计划全部定稿）。下一步：**v2 backlog**（web UI `serve` 模式 / async 重构，见「M3 之后」节与问答日志）。

## 问答日志（问题 → 答复）

1. **vgent 是什么东西？** → 运行时 harness：参考成熟实现（Codex CLI / Claude Code 路线），做一个自己日常能用的通用 agent CLI。
2. **日常拿它干什么？** → 追问社区共识后给出推荐：聚焦「本地第一」——文件操作 + shell 执行 + grep 搜索为 v1 核心，web fetch 可选，浏览器放 v2。写代码和文件杂活收敛到同一组工具。（用户未明确反对，按默认采纳）
3. **模型接入策略？** → **A 方案**：只做 OpenAI-compatible 协议，支持多 provider 配置（base_url / api_key / model）。**主力模型：DeepSeek。**
4. **loop 结构 + 会话存储？** → **交互式 REPL**（多轮对话 + 工具交替，会话可存盘恢复）；会话历史存 **SQLite**（用户拍板，替代推荐的 JSON 明文）。**存储布局扩为：除源码外的一切（配置/会话/日志/缓存）都不放同步盘，放本机 ~/.vgent**（用户确认）。
5. **上下文管理？** → **锚定 hermes-agent（Nous Hermes Agent）的 ContextCompressor 实现**（用户提供本地参考库并指定参照它）。用户纠正：**DeepSeek V4 Flash 上下文是 1M**（修正早前 64K 假设）。方案见决策 8。

## 已定决策

| # | 决策点 | 结论 |
|---|--------|------|
| 1 | 形态 | 运行时 harness / 通用 agent CLI，参考 Codex CLI、Claude Code 路线 |
| 2 | 模型协议 | ✅ 只做 OpenAI-compatible，多 provider 配置（`[providers.<name>]` + `[provider] active` + CLI `--provider`，2026-08-20 落地）；不提前做多协议抽象 |
| 3 | 主力模型 | DeepSeek（国产 OpenAI-compatible 全家桶：Qwen/GLM/Kimi/Ollama 可后补配置） |
| 4 | 平台 | Windows 第一（Git Bash 环境）；shell 层适配 Git Bash/PowerShell，路径处理从第一天写可移植（默认采纳，可改） |
| 5 | v1 工具面 | 文件操作 + shell 执行 + 搜索（grep/ripgrep）；web fetch 可选扩展；浏览器控制 v2（默认采纳，可改） |
| 6 | loop 形态 | **✅ 交互式 REPL**：多轮对话 + 工具调用交替；会话可存盘、可恢复 |
| 7 | 存储布局 | **✅ 除源码外的所有配置与运行数据放本机，不放同步盘**。布局见下「目录规划」；跨机连续性由 HANDOFF.md（同步盘内唯一事实源）承担 |
| 8 | 上下文策略 | **✅ 横向对比综合方案**（各取所长，出处见「参考库」）：① 主干 = hermes-agent：API usage 计 token + 低水位免费剪枝（工具结果一行摘要、清孤儿 tool 对）+ 高水位（默认 75% 可配）「保护头尾 + 中间 LLM 摘要带标记」② + ag2：压缩切点**对齐回合边界**、摘要事件占位（CompactStrategy 可插拔，v1 内置 TailWindow + Summarize 两策略）③ + OpenManus：硬下限兜底（极端超长直接丢最旧）④ + openclaw：三档触发 threshold\|manual\|overflow（v1 做 threshold + **手动 /compact 命令**，用户友好）⑤ SQLite 留全量历史，压缩只影响发送列表；插拔点 = ContextEngine 抽象。**不抄**：responses.compact（DeepSeek 无此 API）、focus_topic/失败冷却/defrag/遥测（v2） |
| 9 | 工具调用协议 | **✅ 原生 tool calling**（JSON Schema 工具定义 + API 层直接返回结构化 tool_calls；执行器容忍空/坏参数，把解析失败回喂模型修正）。全库调研印证：主流实现无一家用文本协议 |
| 10 | 技术栈 | **✅ Python 3.11+ / uv / openai SDK（sync，base_url 指 DeepSeek）/ prompt_toolkit+rich / pytest+ruff**。v1 全 sync（async 重构 v2）；LLM 客户端用 SDK 不用裸 HTTP（用户拍板） |

## 设计思路（社区共识参考，写代码时遵循）

- 核心 loop 自己写（约 50~100 行），**不用 LangChain 类重框架**：调 LLM → 拿 tool_calls → 执行 → 结果塞回消息历史 → 重复
- 工具层：JSON Schema 定义工具 → dispatcher 按工具名映射到本地函数
- 权限模型：起步「读类自动放行 + 写/执行类确认」，后续再分级细化
- 上下文管理：先靠长上下文 + 简单截断/摘要；compaction 视需要再做（DeepSeek 窗口内如何组织，见未决）
- 演进路线：REPL → 工具 → 权限 → 会话持久化 → 多模型 → 界面/记忆

## 目录规划（草案）

```
同步盘 \1_vgent\              ← 只放：源码 + HANDOFF.md + 设计文档
本机 %USERPROFILE%\.vgent\    ← 全部本地，不同步
    ├─ config.toml    （[providers.<name>] 多 provider + active；api_key / api_key_env）
    ├─ sessions\      （SQLite 会话库）
    ├─ logs\
    └─ cache\
```

## 参考库与方法论（本地，跨机接力时同样适用）

`D:\大五\8_GitHub网红项目`（HANDOFF 早前误写「工作即大五」，2026-08-20 修正）——用户学习过的 GitHub 网红 agent 项目合集。**vgent 设计的第一参考来源**。

**方法论（用户拍板）**：不是只参考单一实现，而是**各取所长**——对多个主流实现（openmanus、openai-agents-python、MetaGPT、hermes-agent、crewAI、langgraph、gemini-cli、openclaw 等）做横向对比，综合出「符合当下主流实现、用户友好」的方案；决策记录里注明「此点取自谁」。

已读参考：
- **hermes-agent**（Nous Research）：`agent/context_engine.py`（可插拔 ContextEngine 抽象）+ `agent/context_compressor.py`（压缩实现：免费剪枝→保护头尾→中间 LLM 摘要；1M 窗口下以免费剪枝为主角）
- **横向调研结论（上下文/会话管理，2026-08-20，子代理并行读源码）**：
  - openai-agents-python：Session 抽象 + SQLiteSession（WAL、双表 JSON 存消息）；compaction 走 API 端 responses.compact（**DeepSeek 无此 API，不取**，取其 session 装饰器/阈值钩子模式）；权限：ToolApprovalItem + sticky 放行
  - OpenManus：最简——max_messages=100 硬截断、token 预算超限直接结束（反面教材，取其「硬下限兜底」思路）
  - MetaGPT：BrainMemory LLM 摘要 + 向量长期记忆（v1 不取，v2 记忆可选）
  - crewAI：向量记忆 + 复合打分（v1 不取）；respect_context_window 取 75% 与主流一致
  - langgraph：Checkpointer 快照链 + thread_id + 时间旅行（不取全套，取其 **thread_id 会话主键**概念）
  - ag2：**压缩架构最优**——CompactStrategy 可插拔（TailWindow 零成本 / Summarize LLM 摘要）+ **切点对齐回合边界** + 摘要事件占位（v1 取压缩策略与边界对齐，不取事件溯源）
  - gemini-cli：会话 flag 族（--list-sessions / --resume latest|idx|uuid / --delete-session）；50% 阈值触发 + 保留最近 30%（UX 参考，进分支 5）
  - openclaw：SQLite+JSONL 镜像、**记住上次会话**（tui-last-session）、三档压缩触发（threshold|manual|overflow）、ACP 审批分类器（权限参考，进分支 3）
  - AutoGPT：摘要预算（保留最近 N 条完整 + 更早合并摘要）；权限分层 allow/deny + scope（ONCE/AGENT/WORKSPACE，进分支 3）
  - **主流共识**：① 上下文没有一家「全量硬扛」，都是「免费剪枝打底 + 阈值（50~75%）触发摘要 + 手动兜底」；② 会话全部持久化 + 恢复/列出命令；③ 权限无一缺席，全是分类自动放行 + 分级确认

## 未决问题（下一站从这里继续）

grill 进度：**分支 1 ✅、分支 2 ✅ 全部完成**（决策 1-9 落定：定位/模型/平台/工具面/loop/存储/布局/上下文策略/工具调用）。剩余分支（3 工具与执行、4 技术栈、5 接口与体验）的决策**在框架梳理阶段吸收**，不再单独 grill。

**当前阶段：框架梳理已完成**（用户定的三步走）：
1. 核心组件分解 **✅ 已完成并确认**（见下节：7 组件 + 配置模块；配置不设独立组件、权限 v1 三档）
2. 接口契约 + 数据流 **✅ 已落签名**（见「接口契约 v0.1」；优化推迟：没证据不做深度优化）
3. 具体实现 + **MVP 竖切**：**M0-M2 已 ✅ 完成**（REPL→LLM→SQLite→tools→权限），正推进 **M3 上下文引擎**

**进度**：设计/框架 ✅、M0 ✅、M1 ✅、**M2 ✅**、**M3 ✅**、**M4 ✅**、**M5 ✅ 完成**（2026-08-20，见「构建日志」）——**v1 全部里程碑收官**。M0-M2 已通过一次复核审查（契约/MVP 结论：均可行）。下一步：v2 backlog。

## M3 开工前待办（✅ 2026-08-20 全部完成，M3 可直接开工）

1. **requires-python ≥3.12 + venv ≥3.12.13**（用户拍板 + 本机复核，见 M0 踩坑记录）——`.python-version` 已钉 `3.12.13`，两机各自 `uv sync` 即可（uv 自动下载 3.12.13）。
2. **真实模型联调冒烟 ✅（2026-08-20，本机已配 DEEPSEEK_API_KEY 环境变量）**，验证结果：
   - `include_usage` ✅：每个流末尾都有 usage chunk，`ChatResult.usage` 非 None（含 `reasoning_tokens` 明细）
   - tool_calls 分片 ✅：按位置索引 0 连续发送，`llm.py` 按索引累积成立
   - `reasoning_content` ⚠️→**已修**：deepseek-v4-flash 是思考模式，assistant 消息**必须原样回传** reasoning_content，否则 400（`The reasoning_content in the thinking mode must be passed back to the API`）。已改 `Message`/`store`/`llm` 支持存取与回传（SQLite 加列带 PRAGMA 迁移）；顺带发现 **openai SDK 实际解析到 3.3.1**（pyproject 约束 `>=1.40`），其 `ChoiceDelta` 不声明该字段、只在 `model_extra`，读取逻辑已兼容 1.x 声明字段 / 3.x model_extra 两版
   - 实测副产品：模型会把工具名拼错（`shellread_file`）连错 5 次才自我纠正——「未知工具回喂 + MAX_TOOL_ROUNDS=20」按设计工作，但每轮纠错都烧 token，后续可考虑在错误回喂里给更明确的提示（backlog）；`content=''`（纯思考+tool_calls）的 assistant 消息正常往返
   - 冒烟后测试 **37 passed、ruff 全绿**；「确认交互真终端体验」仍未在真终端验证（headless 冒烟用 ALWAYS 放行），待用户真终端跑一次
3. **M3 设计落点（审查已核对，实现照此）**：engine 挂载在 `agent.py` 的 `list(msgs)` 快照**之前**；TailWindow 剪枝策略先做、Summarize 摘要随后（`/compact` 依赖 Summarize，落 M4）；规划文件 `src/vgent/context.py`（M3 新建）。
4. M3 输入的 Config 已就绪：`context_length=1_000_000`、`threshold_percent=0.75`（高水位）、`prune_percent=0.30`（低水位）、`tail_token_budget=20000`。

## M3 之后 · M4/M5 待办增强（backlog，2026-08-20 审查登记）

- **M4 ✅（2026-08-20 完成）**：Summarize（LLM 摘要）策略 + `/compact`（手动强制压缩：Summarize 中间段，无 summarizer/失败自动退回 TailWindow；压缩结果作发送底稿、SQLite 全量历史不动）+ 状态栏（每轮 tok ↑prompt ↓completion = total、会话累计、压缩次数）+ 会话 title 自动生成（首条用户消息首行 ≤24 字符）+ 新配置 `compact_strategy`（tail|summarize）。zcode 化部分（/list-sessions /delete-session、记住上次会话、全局安装）此前已提前完成。
- **M5 ✅（2026-08-20 完成，见构建日志）**：write_file / search 补齐、LLM 错误退避重试、MAX_TOOL_ROUNDS 安全阀路径测试、tool_calls 按 id 合并防御交错/复用分片、reasoning_content 思考展示（/reasoning + show_reasoning）。`llm.py` 流式累积单测此前已在 M3 待办完成（test_llm.py），本次适配合并逻辑改动。

## 框架梳理 · 核心组件（✅ 2026-08-20 确认）

组件按**消息数据流**排列（每个组件是流上的一个站点）：

```
用户输入 ──▶ ①CLI/REPL ──▶ ②Agent Loop ──▶ ③LLM Provider ──▶ 工具调用
                ▲                │  ▲              │ (usage)        │
                │                │  └──▶⑤上下文引擎◀─┘                ▼
                │                │                     ④会话存储(SQLite)  ⑥工具层
                │                ▼                                     │
             输出 ◀────────── 结果写回 ◀────────── ⑦权限系统 ◀──────────┘
    (流式渲染)                          └── 全部写回 ④ + ⑤剪枝检查
⑧配置层 ── 全局注入（provider/模型/阈值/权限策略）
```

| # | 组件 | 职责要点 | 蓝本/出处 |
|---|------|---------|----------|
| 1 | CLI/REPL 界面层 | 交互输入、流式渲染、命令（/compact /new /resume）、会话切换 | gemini-cli / openclaw（分支 5） |
| 2 | Agent Loop 核心 | 对话状态机：调模型→tool_calls→权限→执行→写回→判定 | openai-agents run_loop |
| 3 | LLM Provider 客户端 | OpenAI-compatible、流式、tool calling 解析、usage 上报 | 决策 2/3/9 |
| 4 | 消息与会话存储 | SQLite 双表 + thread_id，WAL | openai-agents SQLiteSession / langgraph（决策 6/7） |
| 5 | 上下文引擎 | usage 计数、低水位剪枝、边界对齐、TailWindow/Summarize、/compact | 决策 8 综合方案 |
| 6 | 工具层 | JSON Schema 定义 + dispatcher；内置 shell/文件/搜索 | 决策 5（分支 3） |
| 7 | 权限/确认系统 | **v1 三档**：读类自动放行 / 写改类确认 / 执行类确认+sticky 放行（用户拍板） | openclaw 分类器 / openai-agents（分支 3） |
| 8 | 配置模块（非组件） | ~/.vgent/config.toml，**由②启动时加载注入，不设独立组件**（用户拍板） | 决策 7 |

**关键接口契约（✅ 基线已确认，技术栈定后落成签名）**：① usage：组件③ → ⑤ 每轮上报 ② 压缩时机：② 发请求前问⑤ should_compress，压缩只动发送列表、④ 不动 ③ 权限位置：② 拿到 tool_calls 后、⑥ 执行前，过⑦ ④ 流式：③ 流式输出 → ① 渲染，最终消息写回④ ⑤ 工具结果：执行后写回消息列表 + ④，并过⑤ 低水位剪枝检查。

## 接口契约 v0.1（✅ 2026-08-20 用户确认）

```python
# 消息模型（②loop / ③LLM / ④存储 / ⑤引擎 共用）
@dataclass
class Message:
    role: str                          # "system"|"user"|"assistant"|"tool"
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str                     # JSON 原文；容忍坏参数（决策 9）

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

| 契约 | 接口签名 | 接线位置 |
|---|---|---|
| ① usage 上报 | `ContextEngine.update_from_response(usage: Usage)` | ③ 每次 chat 后调用 |
| ② 压缩时机 | `ContextEngine.should_compress() -> bool`；`compress(messages) -> messages` | ② 发请求前问；compress 只动发送列表、④ 不动 |
| ③ 权限位置 | `PermissionSystem.check(tool, args) -> Approval`；`confirm(tool, args) -> ConfirmResult`（M2 细化，见下）；`approve_sticky(tool)` | ② 拿到 tool_calls 后、⑥ execute 前 |
| ④ 流式 | `LLMClient.chat(messages, tools, on_delta=None) -> ChatResult(messages, usage, tool_calls)` | on_delta → ① 渲染；最终消息写回 ④ |
| ⑤ 结果写回 | `ToolRegistry.execute(name, args) -> str`；`ContextEngine.prune_tool_results_only(messages) -> (messages, n)` | 执行后写回列表+④，过低水位检查 |

**M2 细化（2026-08-20）**：③ 的 `confirm` 由 bool 升级为三态 `ConfirmResult`（APPROVE 一次 / ALWAYS 本会话 sticky / REJECT），`ALWAYS` 由 PermissionSystem 自动写入 sticky——满足决策 7「执行类确认+sticky 放行」的 UX。`Approval` 保留 AUTO/NEED_CONFIRM/DENIED（DENIED 用于未知权限档位 / 未来 deny 规则；vgent 内置工具不产生，见 permission.py）。

**Loop 串联（「深度连接」的答案——5 条契约在此全部接线）**：

```python
def run_turn(user_input: str, ctx: SessionContext) -> None:
    msgs = ctx.store.get_history(ctx.session_id) + [Message("user", user_input)]
    while True:
        msgs, _ = ctx.engine.prune_tool_results_only(msgs)          # 契约⑤ 低水位
        if ctx.engine.should_compress():                            # 契约② 高水位
            msgs = ctx.engine.compress(msgs)                        # 只动发送列表
        result = ctx.llm.chat(msgs, ctx.tools.schemas(), on_delta=render)  # 契约④
        ctx.engine.update_from_response(result.usage)               # 契约①
        persist(result.messages, ctx)                               # 契约④ 写回 SQLite
        if not result.tool_calls: return                            # 无工具调用 → 回合结束
        for tc in result.tool_calls:
            if ctx.permissions.check(tc) needs confirm and not confirm(tc):  # 契约③
                persist(reject_msg, ctx); continue
            output = ctx.tools.execute(tc.name, safe_parse(tc.arguments))     # 契约⑤
            persist(Message("tool", output, tool_call_id=tc.id), ctx)
```

## MVP 竖切计划（✅ 2026-08-20 用户确认）

```
M0 骨架      uv init、pyproject、config 加载（tomllib）、日志、目录结构
M1 最小闭环  REPL(rich 渲染) → LLM 流式对话 → SQLite 存取 → /new /resume 可验证
M2 工具+权限 tools 注册 + shell / read_file + 三档权限确认交互（rich prompt）
M3 上下文    ContextEngine：usage 计数 + 低水位剪枝（1M 主角）；高水位摘要随后
M4 UX+命令   ✅ /compact（Summarize）、/list-sessions /delete-session、记住上次会话、状态栏(token 用量)、title 自动生成
M5 收尾      ✅ write_file / search 补齐、LLM 错误重试、测试补强（安全阀路径、tool_calls 按 id 合并、思考展示）——v1 收官
```

项目结构（同步盘 1_vgent 下；运行时数据仍在本机 ~/.vgent）：
```
pyproject.toml        # uv，name=vgent
src/vgent/
  cli.py              # ① REPL + 命令分发
  agent.py            # ② loop（run_turn）
  llm.py              # ③ client（openai SDK sync）
  store.py            # ④ SQLite（双表 + thread_id）
  context.py          # ⑤ engine
  tools.py            # ⑥ registry + 内置工具
  permission.py       # ⑦ 三档
  config.py           # ⑧ 加载 ~/.vgent/config.toml
tests/
HANDOFF.md
```

## 构建日志（里程碑进度，跨机接力看这里）

> 约定：**每个里程碑在独立对话中执行**。开工前：读 HANDOFF（决策表 + 接口契约 v0.1 + MVP 计划）；完成后：在本节追加该里程碑的记录（做了什么、验证结果、遗留问题），另一台机器据此续下一个里程碑。

- **M0（骨架）— ✅ 2026-08-20 完成**。创建：pyproject.toml（uv/hatchling；deps: openai/prompt-toolkit/rich；dev: pytest/ruff）、.python-version=**3.12**（⚠️ 有意的，别改回 3.11，见下）、.gitignore、src/vgent/{__init__,config,cli,__main__}.py、tests/test_config.py（4 例）、README.md。验证：`uv run vgent` 正常输出、pytest 4 passed、ruff 全绿。venv 在本机 `%USERPROFILE%\.vgent\venv-vgent`（命令前设 `UV_PROJECT_ENVIRONMENT`；两机各自 uv sync）。
  **踩坑记录（跨机重要）**：项目路径含中文 → Python 3.11（本机 3.11.2）的 site.py 用 GBK 读 editable 安装生成的 `.pth`（内容含中文路径）→ UnicodeDecodeError，连 `PYTHONUTF8=1` 也无效（Windows frozen site）。**解法：venv 用 Python 3.12.13+**。⚠️ 2026-08-20 本机复核修正：**3.12.2 的 site.py 仍按 locale 编码（中文 locale=GBK）读 `.pth`，同样必崩**；3.12.13+/3.13 才改为 utf-8-sig 优先（早前「3.12 已修复」的结论只对非 GBK locale 成立）。`.python-version` 已钉 `3.12.13`，另一台机器直接 uv sync 即可（uv 自动下载 3.12.13）。
  **遗留**：DeepSeek API 的 model 名（deepseek-v4-flash）与 context_length=1M 待 M1 联调验证。
- **M1（最小闭环）— ✅ 2026-08-20 完成**。新增：`messages.py`（Message/ToolCall/Usage，项目结构草案的补充）、`store.py`（SQLite 双表+WAL+thread_id）、`llm.py`（openai SDK sync 流式 + include_usage）、`agent.py`（run_turn 按契约④⑤接线，用户消息与 assistant 消息都落库）、`cli.py` 重写为 REPL（rich + 启动会话选择器 + /new /resume /help /exit）；tests 新增 test_store / test_messages / test_agent（共 11 例）。
  验证：pytest 11 passed、ruff 全绿；冒烟（管道输入）：启动→选择器→新建→对话（无 key 时 401 优雅报错不崩溃）→/new→/help→/exit 全程 exit 0；/resume 恢复路径 OK。
  **DeepSeek 核对 ✅**：官网确认 `deepseek-v4-flash`（DeepSeek-V4-Flash-0731）存在，上下文 1M、最大输出 384K——config 默认值无需改。
  **踩坑（跨机重要）**：① prompt_toolkit 在 Git Bash/mintty/管道输入下构造失败（NoConsoleScreenBufferError）→ `cli._make_prompter()` 自动退回 `input()`；完整交互（多行编辑/历史）用 Windows Terminal 或 cmd 运行 ② openai SDK 构造时要求非空 api_key → LLMClient 用占位 key（sk-vgent-missing-key）让 401 发生在请求时被 REPL 兜底 ③ httpx 的 INFO 日志压到 WARNING 防污染输出。
  **遗留**：本机无 DEEPSEEK_API_KEY 且无 config.toml，**真实模型调用未联调过**（401 路径已验证）。M2 开工时若有 key，建议先做一次真实对话冒烟。
- **M2（工具+权限）— ✅ 2026-08-20 完成**。新增 `tools.py`（ToolSchema/Tool 注册表 + 内置 shell/read_file，输出硬上限 10K 防病态输出）、`permission.py`（三档：read 自动 / write+exec 确认 / 未知档 DENIED；ConfirmResult 三态 + ALWAYS 自动 sticky；无确认交互时默认拒绝 = headless 安全默认）；`agent.py` 升级为完整工具循环（chat→tool_calls→权限→执行→写回→再 chat；MAX_TOOL_ROUNDS=20 安全阀，超限强制收尾；`_safe_parse` 容忍坏参数回喂；`on_tool` 回调）；`cli.py` 注入 default_tools + rich 确认交互（y 一次 / a 本会话 / n 拒绝）+ 工具状态行。
  验证：pytest **31 passed**、ruff 全绿；冒烟（管道输入）启动→新建→/help→/exit 全程 exit 0。
  **踩坑**：① read_file 行号双重递增（enumerate(start) 又 +1）——已修；② run_turn 把可变 msgs 直接传给 LLM，随后 extend 污染「发给模型的历史」快照——已改传 `list(msgs)` 快照（测试桩暴露的真 bug）。
  **遗留**：真实模型联调仍未做（本机无 DEEPSEEK_API_KEY）——shell/read_file 的真模型驱动调用、确认交互真终端体验，待有 key 的机器冒烟。下一里程碑 **M3 上下文**：ContextEngine（usage 计数 + 低水位剪枝；高水位摘要随后）。
- **M3 开工前待办 — ✅ 2026-08-20 完成**（见上节：真实模型冒烟 + reasoning_content 回传修复）。
  - 修复内容：`messages.py` Message 加 `reasoning_content` 字段（`to_openai` 原样回传）；`store.py` messages 表加列（PRAGMA 迁移兼容旧库）；`llm.py` 流式累积 reasoning_content（兼容 openai 1.x 声明字段 / 3.x `model_extra` 两版）；新增 `tests/test_llm.py`（假 chunk 流式累积 4 例）+ messages/store 各 1 例。
  - 验证：测试 31 → **37 passed**、ruff 全绿；真实模型冒烟（shell + read_file 各一次）端到端通过。
- **配置增强：多 provider（2026-08-20，M3 前小步）**：`config.py` 支持 `[providers.<name>]` 定义 + `[provider] active` 选择 + CLI `--provider <name>` 覆盖；每 provider 可独立设 `api_key_env`（**默认空 = 只用文件里的 api_key**——冒烟实测踩坑：默认值曾写死 `DEEPSEEK_API_KEY`，本机已设该环境变量导致 DeepSeek key 串到 tokenrhythm 报 401，修正为显式声明）；旧式单 `[provider]` 字段写法向后兼容。本机 `~/.vgent/config.toml` 已建：**tokenrhythm 激活**（`deepseek-v4-flash-0731` @ `https://tokenrhythm.studio/v1`，key 在文件里），deepseek 备选（env `DEEPSEEK_API_KEY`）。
  验证：**43 passed**、ruff 全绿；tokenrhythm 真实冒烟 ✅（include_usage、tool_calls、reasoning_content——**该网关同为思考模式**，上一条的 reasoning_content 回传修复通用；curl 直连也 200，非流式响应含 `cost_cny`/`billing_pending` 计费字段）。
- **zcode 化（M4 前置部分，2026-08-20）**：`cli.py` 增加 flag 族 `--new` / `--resume[ID|N|last]` / `--list-sessions` / `--delete-session` / `--version`（argparse）+ REPL 命令 `/list` `/delete`；启动选择器加「0=上次会话」，记住上次会话落 `~/.vgent/last_session`（会话被删自动失效）；`store.list_sessions` 排序加 `rowid DESC` 兜底（同秒创建确定性）。分发：`uv build` 出 wheel（dist/ 已 gitignore），`uv tool install .` 全局命令 `vgent` 可用（`~/.local/bin/vgent`；**代码更新后需 `uv tool install . --force` 重装**）。
  验证：**48 passed**、ruff 全绿；`vgent --version`、`vgent --list-sessions`、`vgent --resume` 解析实测通过。
- **M3（上下文引擎）— ✅ 2026-08-20 完成**：新建 `src/vgent/context.py`（ContextEngine，契约①②⑤：`update_from_response` usage 计数 / `should_compress`+`compress` 高水位 / `prune_tool_results_only` 低水位）。
  - 实现要点（蓝本 hermes-agent `context_compressor.py`，本机参考库已核对）：TailWindow 策略——保护首条 + 尾部 token 预算（`tail_token_budget`），中间整体丢弃并插入 **system 标记消息**；切点对齐回合边界（不切分 `assistant(tool_calls)+tool*` 对）；低水位剪枝把长工具结果压一行摘要 + 清孤儿 tool 对（尾部按**消息条数**保护 6 条——hermes 经验：1M 窗口下按 token 保护会剪不掉）；硬下限兜底丢最旧（OpenManus）；token 估算用本地启发式（~3 字符/token）经 API usage 校准。
  - `agent.py` 接线：`SessionContext.engine`（默认 factory，旧测试无感）+ `run_turn` 循环内按契约 ⑤→②→④→① 顺序调用；`MAX_TOOL_ROUNDS` 溢出路径的 final chat 也报 usage。`cli.py` 用 `cfg.provider.context_length + cfg.context` 构造。
  - 测试：新增 `tests/test_context.py` 9 例（usage 驱动高水位、剪枝摘要/孤儿清理、TailWindow 头尾+标记、tool 对不切分、硬下限保留最后一条）+ `test_agent.py` 集成 2 例（run_turn 里剪枝/压缩生效且 store 全量不动）→ 48 → **59 passed**、ruff 全绿。
  - 真实冒烟 ✅（tokenrhythm）：预置 36 条历史逼压缩 → chat 发送列表压到 16 条、**含 system 标记且 API 接受无 400**；shell 工具调用正常。
  - **已知边界（记入 M4 前评估）**：启发式估算对中文偏低——冒烟实测发送 prompt_tokens=1273 略超 context_length=1200 但无报错；1M 窗口 + 0.75 阈值余量巨大，可接受；usage 校准在每次 chat 后生效。
- **M4（UX+命令）— ✅ 2026-08-20 完成**。Summarize 策略 + /compact + 状态栏 + title 自动生成：
  - `context.py`：`compress(messages, strategy=None, force=False)`——strategy 缺省取新配置 `cfg.compact_strategy`（"tail"|"summarize"）；可注入 `summarizer`（cli 注入 LLM 摘要器）；summarize 分支把中间段交给 LLM 压成「【历史摘要（原 N 条）】」消息，无 summarizer/异常/空摘要自动退回 TailWindow 标记；`force=True` 跳过水位检查（/compact 手动触发）；新增 `compacted` 字段作为发送底稿。
  - `agent.py`：首条用户消息自动生成会话标题（首行 ≤24 字符，`store.update_title`）；run_turn 优先用 `engine.compacted` 作为发送底稿（只影响发送列表，SQLite 全量不动）。
  - `cli.py`：`/compact` 命令（Summarize 强制压缩 + 底稿 + 条数反馈）；每轮回答后状态栏 `tok ↑prompt ↓completion =total；会话累计；压缩次数`；`/new` `/resume` 清底稿与累计。
  - 测试：test_context +4（summarize 生效 / 无 summarizer 退回 / 摘要异常退回 / force 绕过水位）、test_store +1（update_title）、test_agent +2（title 首条生成不覆盖、compacted 底稿生效且 store 不动）、test_cli +1（/compact 处理器）→ 59 → **67 passed**、ruff 全绿。
  - 真实冒烟 ✅（tokenrhythm）：60 条历史 → `/compact` 压成 8 条（头 + 【历史摘要（原 53 条）】+ 尾），模型**仅凭摘要即正确回答**模块分工问题；压缩底稿真实对话无 400（reasoning_content 回传复用）。默认 `tail_token_budget=20000` 下纯文本历史会被完整保护（/compact 报「无可压缩内容」）——符合设计；/compact 在工具输出撑大的历史里才真正生效。
  - 全局命令已 `uv tool install . --force` 重装（含 M4 代码）。
  - **遗留（M5 参考）**：`compact_strategy="summarize"` 自动触发已实现但未在真实长会话连续验证；状态栏/确认交互真终端体验待用户跑一次。
- **M5 ✅（2026-08-20 完成，v1 收官）**：write_file / search 工具补齐 + LLM 错误重试 + 测试补强：
  - `tools.py`：+`write_file`（write 档，overwrite/append、自动建目录）+`search`（read 档，递归正则，自动跳过 .git/node_modules/.venv 等噪音目录，单行/条数/总输出三重上限）——补全决策 5 的 v1 工具面（文件操作 + shell + 搜索）。
  - `llm.py`：① 可重试错误（429/5xx/连接/超时）指数退避重试（`max_retries=0` 关 SDK 内建避免叠加；确定性错误 401/400 不重试直接抛）；② tool_calls 合并改为「有 id 按 id 归槽保序、无 id 续片按 index 归属、同 index 新 id 自动开新槽」——**修了个真 bug**：流式分片的工具序号在 `.index` 字段而非 chunk 数组位置（数组常单元素，enumerate 位置不可靠），并行工具调用会拼错；③ +`on_reasoning` 回调。
  - `agent.py`/`cli.py`/`config.py`：`show_reasoning` 配置（config.toml 顶层）+ REPL `/reasoning` 切换 + dim 样式流式渲染思考内容（默认关）。
  - 测试：**67 → 81 passed**（+14：write_file/search 6、tool_calls 交错合并/同 index 换 id/on_reasoning 3、重试成功/放弃/不重试 3、MAX_TOOL_ROUNDS 安全阀路径 1、show_reasoning 1）、ruff 全绿；tokenrhythm 真机冒烟 ✅（write_file 18 字符落盘 + search 命中 + 多轮工具循环 reasoning 回传正常）。全局命令已 `uv tool install . --force` 重装。
  - **遗留（用户侧）**：状态栏/确认交互/思考展示的真终端体验待用户跑一次；`compact_strategy="summarize"` 自动触发未在真实长会话连续验证。
- **真机首跑修复（2026-08-20）**：用户真终端首跑（PowerShell + tokenrhythm）验证三体验全部正常，发现并修复 3 个 bug：
  - **shell 工具找不到 bash**（真机阻塞）：Git 装在非标准路径 `D:\git\Git`（注册表 `HKLM\SOFTWARE\GitForWindows` InstallPath 有值），原 `_resolve_shell` 只查 `C:\Program Files\Git` 等四个写死路径，且 PowerShell PATH 无 Git → 增加两级兜底：注册表 InstallPath + PATH 里 `git` 路径向上推断根目录。
  - **httpx2 日志噪音**：openai SDK 3.x 的日志器名是 `httpx2`（非 `httpx`），`setup_logging` 只压了 httpx → 同步压 WARNING。
  - **banner 吞 provider 名**：`[{cfg.provider.name}]` 被 rich 当样式标签吞掉（显示成 `—  model`）→ 转义 `\[name\]`；顺带会话标题/确认参数/工具输出行加 rich 转义（`markup=False` / `escape`），防含 `[]` 的用户文本被当样式解析。
  - 测试 81 → **83 passed**（+2：git 根推断、注册表解析）、ruff 全绿。
  - 体验确认 ✅：确认交互（y 一次 / a sticky / 拒绝回喂）、状态栏（`tok ↑↓=total；会话累计`，复测累计 1361+1462=2823 正确）、思考展示（`/reasoning` dim 渲染；trivial 任务模型可能不输出思考，属正常）、title 自动生成、`/list`、`/compact` 短历史预期行为、**`vgent --resume` 实测直接恢复上次会话**。使用注意：`vgent --resume` 是 shell 命令，须先 `/exit` 再在终端执行（在 REPL 里输入会被当消息发给模型，实测烧 ~5.9K tokens）。
- **v2 backlog（未排期）**：web UI（`serve` 模式，FastAPI + SSE，复用 SessionContext；async 重构——决策 10 预留）；会话导出；权限策略配置化。

## 用法约定

- 每次跨设备开工/收工：在本文件更新「最近更新」日期、追加问答日志、刷新决策表
- 决策从「推荐」变「拍板」时，在表格里标 ✅
- 里程碑分对话执行：每完成一步回写「构建日志」，另一台机器读 HANDOFF 后从下一步接着做
