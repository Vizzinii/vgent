"""M8 测试：episodic 记忆（JSONL 存取 + LLM 摘要）。"""
from __future__ import annotations

from types import SimpleNamespace

from vgent.memory.episodic import EpisodicMemory, summarize
from vgent.messages import Message


def test_add_and_reopen(tmp_path) -> None:
    path = tmp_path / "episodic.jsonl"
    EpisodicMemory(path).add("优化 git 仓库", "扫描了 3 个仓库，发现 1 个问题", "sid1", "优化 git 仓库性能")
    mem = EpisodicMemory(path)  # 重新打开（模拟重启）
    assert mem.count() == 1
    e = mem.search("git")[0]
    assert e.topic == "优化 git 仓库"
    assert e.session_id == "sid1"
    assert e.title == "优化 git 仓库性能"


def test_search_case_insensitive_and_limit(tmp_path) -> None:
    mem = EpisodicMemory(tmp_path / "m.jsonl")
    mem.add("Python 重构", "a", "s1", "t1")
    mem.add("前端优化", "b", "s2", "t2")
    mem.add("Python 性能", "c", "s3", "t3")
    assert len(mem.search("python")) == 2
    assert len(mem.search("python", limit=1)) == 1
    assert mem.search("nope") == []
    assert mem.search("") == []


def test_has_session_and_bad_lines_skipped(tmp_path) -> None:
    path = tmp_path / "m.jsonl"
    mem = EpisodicMemory(path)
    mem.add("t", "s", "sid9", "title")
    path.write_text("not json\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    assert mem.count() == 1  # 坏行跳过不阻断
    assert mem.has_session("sid9")
    assert not mem.has_session("other")


def test_list_recent_returns_last(tmp_path) -> None:
    mem = EpisodicMemory(tmp_path / "m.jsonl")
    for i in range(15):
        mem.add(f"topic{i}", f"summary{i}", f"s{i}", f"t{i}")
    recent = mem.list_recent(10)
    assert len(recent) == 10
    assert recent[-1].topic == "topic14"


def test_missing_file_empty(tmp_path) -> None:
    mem = EpisodicMemory(tmp_path / "nope.jsonl")
    assert mem.count() == 0
    assert mem.search("x") == []
    assert mem.list_recent() == []


def test_summarize_returns_text() -> None:
    class FakeLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            return SimpleNamespace(
                messages=[Message("assistant", "做了 X，得出结论 Y，遗留事项是 Z，下一步待办 W")]
            )

    assert summarize([Message("user", "hi")], FakeLLM(), "主题") == (
        "做了 X，得出结论 Y，遗留事项是 Z，下一步待办 W"
    )


def test_summarize_empty_on_exception() -> None:
    class BoomLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            raise RuntimeError("api down")

    assert summarize([Message("user", "hi")], BoomLLM(), "x") == ""


def test_summarize_empty_on_blank() -> None:
    class EmptyLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            return SimpleNamespace(messages=[Message("assistant", "   ")])

    assert summarize([Message("user", "hi")], EmptyLLM(), "x") == ""


def test_summarize_falls_back_to_reasoning_when_content_short() -> None:
    """M8 跟进：正文只吐一句碎片（deepseek 思考模式踩坑）→ 退回思考流。"""

    class FragmentedLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            return SimpleNamespace(
                messages=[
                    Message(
                        "assistant",
                        "好的，我再试一次。",  # 碎片正文（< 20 字符）
                        reasoning_content="扫描了 tools.py，总结出 4 个内置工具及权限档位",
                    )
                ]
            )

    assert summarize([Message("user", "hi")], FragmentedLLM(), "x") == (
        "扫描了 tools.py，总结出 4 个内置工具及权限档位"
    )


def test_summarize_rejects_short_fragment() -> None:
    """M8 跟进：正文与思考流都过短（纯碎片）→ 判失败返回空，不写入记忆。"""

    class JunkLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            return SimpleNamespace(
                messages=[
                    Message("assistant", "好的，我再试一次。", reasoning_content="嗯")
                ]
            )

    assert summarize([Message("user", "hi")], JunkLLM(), "x") == ""


def test_summarize_strips_analysis_block() -> None:
    """P4：模型输出 <analysis>+<summary> 结构化 → 只存 <summary>（草稿不进记忆）。"""

    class StructuredLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            return SimpleNamespace(
                messages=[
                    Message(
                        "assistant",
                        "<analysis>内部草稿，不进记忆</analysis>\n"
                        "<summary>扫描了 3 个文件，结论是升级依赖，遗留：测试未跑</summary>",
                    )
                ]
            )

    assert summarize([Message("user", "hi")], StructuredLLM(), "x") == (
        "扫描了 3 个文件，结论是升级依赖，遗留：测试未跑"
    )


def test_summarize_prefers_summary_over_reasoning() -> None:
    """P4：正文含 <summary> 时优先取正文（即使思考流更长/更完整）。"""

    class BothLLM:
        def chat(self, messages, tools=None, on_delta=None, on_reasoning=None):
            return SimpleNamespace(
                messages=[
                    Message(
                        "assistant",
                        "<summary>正文里的结构化摘要：扫描了三个文件，结论是升级依赖</summary>",
                        reasoning_content="思考流里的更长的内容，但正文已有结构化摘要",
                    )
                ]
            )

    assert summarize([Message("user", "hi")], BothLLM(), "x") == (
        "正文里的结构化摘要：扫描了三个文件，结论是升级依赖"
    )


# -- P5：记忆按项目隔离 ---------------------------------------------------------


def test_project_field_and_filter(tmp_path) -> None:
    mem = EpisodicMemory(tmp_path / "m.jsonl")
    mem.add("A 项目重构", "摘要A", "s1", "A", project="proj-a")
    mem.add("B 项目优化", "摘要B", "s2", "B", project="proj-b")
    # 指定项目：只在该项目内检索（自动回忆防串味）
    assert [e.topic for e in mem.search("项目", project="proj-a")] == ["A 项目重构"]
    # project=None：跨项目（/recall 显式检索）
    assert len(mem.search("项目")) == 2
    # 反向匹配仍生效（用户消息提到 topic）
    assert len(mem.search("继续上次那个 A 项目重构")) == 1


def test_project_default_is_cwd_basename(tmp_path) -> None:
    import os
    from pathlib import Path

    mem = EpisodicMemory(tmp_path / "m.jsonl")
    e = mem.add("主题", "摘要", "s1", "t")
    assert e.project == Path(os.getcwd()).name


def test_legacy_entry_without_project(tmp_path) -> None:
    """旧 JSONL（无 project 字段）照常加载；项目过滤时不串味。"""
    mem = EpisodicMemory(tmp_path / "m.jsonl")
    mem.path.write_text(
        '{"ts": "2026-01-01", "session_id": "s", "title": "t", "topic": "旧主题", "summary": "旧摘要"}\n',
        encoding="utf-8",
    )
    hits = mem.search("旧主题")
    assert len(hits) == 1 and hits[0].project == ""
    assert mem.search("旧主题", project="proj-a") == []


def test_concurrent_add_no_line_loss(tmp_path) -> None:
    """并发安全修复：多线程同时 add 不丢行/不坏行（Web 多线程真实路径）。"""
    import threading

    mem = EpisodicMemory(tmp_path / "m.jsonl")

    def worker(i: int) -> None:
        for j in range(10):
            mem.add(f"t{i}-{j}", f"s{i}-{j}", f"sid{i}-{j}", f"title{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert mem.count() == 80  # 无丢行（坏行会被 from_line 跳过，count 即有效行数）
    assert len({e.topic for e in mem.search("t", limit=100)}) == 80  # 无重复/无坏行
