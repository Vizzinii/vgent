"""全方位测试方案 v2 · D 组：双端一致性（cli _dispatch_command vs web run_command）。

同一初始状态分别跑两端命令，断言副作用（store/文件/内存态）一致——
允许输出文本格式不同。这是「命令层合并」里程碑（B-2）的前置基线。
已知声明差异（标注除外）：/restore 在 CLI 有编号/命名档确认交互，Web 直接执行。
"""
from __future__ import annotations

import io

from rich.console import Console

from vgent.agent import SessionContext
from vgent.cli import _dispatch_command
from vgent.config import PermissionRules
from vgent.llm import ChatResult
from vgent.memory.episodic import EpisodicMemory
from vgent.memory.store import MemoryFileStore
from vgent.messages import Message, Usage
from vgent.permission import PermissionSystem
from vgent.snapshot import SnapshotStore
from vgent.store import SessionStore
from vgent.task import TaskPlan, TaskStep
from vgent.web.server import SessionHub, run_command

_SUMMARY_REPLY = "<summary>这是一段足够长的结构化会话摘要，覆盖了决策与遗留事项，用于双端一致性测试。</summary>"


class _SummaryLLM:
    """summarize 用：返回结构化摘要正文。"""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
        self.calls += 1
        return ChatResult(messages=[Message("assistant", _SUMMARY_REPLY)], usage=Usage(10, 5, 15))


class _Env:
    """一端的完整环境（cli 或 web 各建一个，初始状态相同）。"""

    def __init__(self, root, with_plan=False, with_memory=False, with_snap=False) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.store = SessionStore(root / "t.db")
        self.sid = self.store.create_session()
        self.store.add_messages(
            self.sid,
            [
                Message("user", "第一个问题"),
                Message("assistant", "第一个回答"),
            ],
        )
        if with_plan:
            plan = TaskPlan([TaskStep("做甲", "done"), TaskStep("做乙", "pending")])
            self.store.upsert_plan_message(self.sid, plan.plan_message().content)
        if with_memory:
            ms = MemoryFileStore(root, root)
            ms.ensure_layout()
            ms.atomic_write("MEMORY.md", "v1\n# MEMORY\n\n- 已有条目：测试基线")
            ms.atomic_write("memory_summary.md", "v1\n# Summary\n\n- 测试基线摘要")
            ms.append_raw("\n---\n## 旧流水\n- 旧 bullet\n")
            (ms.root / "rollout_summaries").mkdir(parents=True, exist_ok=True)
            (ms.root / "rollout_summaries" / "old-1.md").write_text("# 旧 rollout\n", encoding="utf-8")
            self.memory_store = ms
            self.memory = EpisodicMemory(root / "episodic.jsonl")
        else:
            self.memory_store = None
            self.memory = None
        if with_snap:
            (root / "f.txt").write_text("CURRENT", encoding="utf-8")
            snaps = SnapshotStore(root / "ck", root)
            snaps.note_before_write("f.txt")
            (root / "f.txt").write_text("CHANGED", encoding="utf-8")
            snaps.seal_turn()
            self.snaps = snaps
        else:
            self.snaps = None
        # config.toml（/allow 持久化用）
        (root / "config.toml").write_text(
            '[permissions]\nask = ["read_file"]\n', encoding="utf-8"
        )
        self.ctx = SessionContext(
            session_id=self.sid,
            store=self.store,
            llm=_SummaryLLM(),
            permissions=PermissionSystem(rules=PermissionRules(ask=["read_file"])),
            plan=TaskPlan([TaskStep("做甲", "done"), TaskStep("做乙", "pending")]) if with_plan else None,
            memory=self.memory,
            memory_file_store=self.memory_store,
            data_dir=root,
            snapshots=self.snaps,
        )
        self.hub = SessionHub(self.ctx)  # web 侧入口
        self.console = Console(file=io.StringIO(), force_terminal=False)
        self.last_path = root / "last_session"
        self.tokens = {"n": 0}

    def cli(self, text: str) -> bool:
        return _dispatch_command(text, self.ctx, self.console, lambda ps: "", self.last_path, self.tokens)

    def web(self, text: str) -> str:
        return run_command(text, self.hub)

    def close(self) -> None:
        self.store.close()


def _dual(tmp_path, **kw) -> tuple[_Env, _Env]:
    return _Env(tmp_path / "cli", **kw), _Env(tmp_path / "web", **kw)


def _plan_messages_in_store(env: _Env) -> list[str]:
    return [m.content for m in env.store.get_history(env.sid) if "[vgent-plan]" in m.content]


# -- /plan 族 ---------------------------------------------------------------


def test_dual_plan_new_and_redo_clear(tmp_path) -> None:
    """双端 /plan new 与 /plan redo：计划消息清空 + ctx.plan=None。"""
    for cmd in ("/plan new", "/plan redo"):
        cli, web = _dual(tmp_path / cmd[1:].replace(" ", "_"), with_plan=True)
        try:
            assert cli.cli(cmd) is True
            assert "已清除" in web.web(cmd)
            assert _plan_messages_in_store(cli) == []
            assert _plan_messages_in_store(web) == []
            assert cli.ctx.plan is None and web.ctx.plan is None
        finally:
            cli.close()
            web.close()


def test_dual_plan_renew_keeps(tmp_path) -> None:
    """双端 /plan renew：参数不精确匹配 new/redo → 不清计划（F9）。"""
    cli, web = _dual(tmp_path, with_plan=True)
    try:
        assert cli.cli("/plan renew") is True
        web.web("/plan renew")
        assert len(_plan_messages_in_store(cli)) == 1
        assert len(_plan_messages_in_store(web)) == 1
        assert cli.ctx.plan is not None and web.ctx.plan is not None
    finally:
        cli.close()
        web.close()


# -- /remember 族 -----------------------------------------------------------


def test_dual_remember_no_args_usage_only(tmp_path) -> None:
    """双端 /remember 无参：提示用法，不调 LLM、不落记忆。"""
    cli, web = _dual(tmp_path, with_memory=True)
    try:
        before_cli = len(cli.store.get_history(cli.sid))
        assert cli.cli("/remember") is True
        web.web("/remember")
        assert cli.ctx.llm.calls == 0 and web.ctx.llm.calls == 0
        assert len(cli.store.get_history(cli.sid)) == before_cli
        assert web.ctx.memory.count() == 0
    finally:
        cli.close()
        web.close()


def test_dual_remember_with_topic(tmp_path) -> None:
    """双端 /remember <主题>：LLM 摘要落 episodic，条数与内容一致。"""
    cli, web = _dual(tmp_path, with_memory=True)
    try:
        assert cli.cli("/remember 测试主题") is True
        web.web("/remember 测试主题")
        assert cli.ctx.memory.count() == 1 and web.ctx.memory.count() == 1
        e_cli = cli.ctx.memory.list_recent()[0]
        e_web = web.ctx.memory.list_recent()[0]
        assert e_cli.topic == e_web.topic == "测试主题"
        assert e_cli.summary == e_web.summary  # 同一 LLM 输出 → 摘要一致
    finally:
        cli.close()
        web.close()


# -- /memory 族 -------------------------------------------------------------


def test_dual_memory_grep_and_clear(tmp_path) -> None:
    """双端 /memory grep（读）与 /memory clear（清空重建 + invalidate 语义）。"""
    cli, web = _dual(tmp_path, with_memory=True)
    try:
        g1 = cli.cli("/memory grep 基线")
        assert g1 is True
        g2 = web.web("/memory grep 基线")
        # 两端都命中同一 MEMORY 条目（输出文本允许不同，内容都含基线条目）
        assert "已有条目" in cli.console.file.getvalue()
        assert "已有条目" in g2
        # clear：两侧文件层副作用一致（回空模板 + rollout 清空）
        assert cli.cli("/memory clear") is True
        web.web("/memory clear")
        for env in (cli, web):
            assert env.memory_store.summary_is_placeholder()
            assert not env.memory_store.memory_has_entries()
            assert list((env.memory_store.root / "rollout_summaries").glob("*.md")) == []
            assert "旧 bullet" not in (env.memory_store.root / "raw_memories.md").read_text(encoding="utf-8")
    finally:
        cli.close()
        web.close()


# -- /allow ---------------------------------------------------------------


def test_dual_allow_persists(tmp_path) -> None:
    """双端 /allow <工具>：sticky + rules.allow + config.toml 写入一致。"""
    cli, web = _dual(tmp_path)
    try:
        assert cli.cli("/allow write_file") is True
        web.web("/allow write_file")
        for env in (cli, web):
            assert "write_file" in env.ctx.permissions.rules.allow
            text = (env.root / "config.toml").read_text(encoding="utf-8")
            assert 'write_file' in text and 'ask = ["read_file"]' in text  # ask 保留
        # 两端 config.toml 语义一致（allow 单例相同）
        import tomllib

        t1 = tomllib.loads((cli.root / "config.toml").read_text(encoding="utf-8"))
        t2 = tomllib.loads((web.root / "config.toml").read_text(encoding="utf-8"))
        assert t1["permissions"] == t2["permissions"]
    finally:
        cli.close()
        web.close()


# -- /snapshot 与 /restore --------------------------------------------------


def test_dual_snapshot_creates_named(tmp_path) -> None:
    """双端 /snapshot 名：命名档文件创建一致。"""
    cli, web = _dual(tmp_path, with_snap=True)
    try:
        assert cli.cli("/snapshot 档一") is True
        web.web("/snapshot 档一")
        for env in (cli, web):
            named = list((env.root / "ck" / "snapshots" / "named").glob("*.json"))
            assert len(named) == 1
    finally:
        cli.close()
        web.close()


def test_dual_restore_last_and_undo(tmp_path) -> None:
    """双端 /restore last 与 /restore undo：文件恢复一致（last/undo 免确认，无双端差异）。"""
    cli, web = _dual(tmp_path, with_snap=True)
    try:
        assert cli.cli("/restore last") is True
        web.web("/restore last")
        assert (cli.root / "f.txt").read_text(encoding="utf-8") == "CURRENT"
        assert (web.root / "f.txt").read_text(encoding="utf-8") == "CURRENT"
        # 再改一次 → undo 回到恢复前
        for env in (cli, web):
            (env.root / "f.txt").write_text("AGAIN", encoding="utf-8")
        assert cli.cli("/restore undo") is True
        web.web("/restore undo")
        assert (cli.root / "f.txt").read_text(encoding="utf-8") == "CHANGED"
        assert (web.root / "f.txt").read_text(encoding="utf-8") == "CHANGED"
    finally:
        cli.close()
        web.close()
