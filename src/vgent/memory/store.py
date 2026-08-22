"""M12-C 记忆文件存储层：只负责路径与文件读写，不调 LLM。

思路来源（只学思路不抄代码）：
- xcode-py MemoryStore（src/memory/store.py）：memory_summary.md（注入 system 的短总览）
  + MEMORY.md（可搜索注册表）+ raw_memories.md（stage1 追加流水）+ rollout_summaries/；
  原子写（tmp+rename）、resolve_rel 防穿越、clear 重建空模板；
- claude memdir：索引与主题分层、summary 只注入截断预览。

目录布局（本机 ~/.vgent/memory/projects/<project_key>/，不进同步盘，决策 7）：
  memory_summary.md     # 极短总览；注入 system；首行 v1
  MEMORY.md             # 主题注册表（可搜索；stage2 维护）
  raw_memories.md       # stage1 追加的原始 bullets 流水
  rollout_summaries/    # 单轮摘要文件，MEMORY 可引用其相对路径

与 episodic.jsonl 的关系：jsonl 仍是「记忆条目」真相（/remember /recall /memories），
本 store 是「项目级浓缩视图」的磁盘层，由两阶段管线（pipeline.py）维护。
"""
from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

SUMMARY_NAME = "memory_summary.md"
MEMORY_NAME = "MEMORY.md"
RAW_NAME = "raw_memories.md"
ROLLOUTS_DIR = "rollout_summaries"

SUMMARY_INJECT_LIMIT = 4000  # 注入 system 的 summary 截断
READ_LIMIT = 12000  # read_rel 默认截断（防超大文件撑爆工具输出）

_EMPTY_SUMMARY = """\
v1
# Memory Summary

（尚无长期记忆）

## What's in Memory
- （空）
"""

_EMPTY_MEMORY = """\
v1
# MEMORY

按主题组织的项目长期记忆注册表。模型按需检索本文件；细节见 rollout_summaries/。
"""

_PROJECT_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")


def project_key(workspace: Path) -> str:
    """项目键 = 工作区顶层目录名（与 episodic.current_project 同口径，P5 隔离语义）。

    目录名做安全化（可落盘）；空名兜底 "workspace"。
    """
    name = _PROJECT_KEY_RE.sub("-", Path(workspace).resolve().name).strip("-").lower()
    return name or "workspace"


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class MemoryFileStore:
    """绑定「一个 workspace」的 memories 根目录上的全部文件操作。"""

    def __init__(self, data_home: Path, workspace: Path) -> None:
        self.data_home = data_home
        self.workspace = Path(workspace).resolve()
        # 与 episodic 的 project 字段同口径：cwd 顶层目录名
        self.root = data_home / "memory" / "projects" / project_key(self.workspace)

    # -- 布局 ---------------------------------------------------------------

    def ensure_layout(self) -> None:
        """缺啥补啥：目录 + 空模板文件，幂等可重复调用。"""
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / ROLLOUTS_DIR).mkdir(parents=True, exist_ok=True)
        summary = self.root / SUMMARY_NAME
        memory = self.root / MEMORY_NAME
        raw = self.root / RAW_NAME
        if not summary.is_file():
            summary.write_text(_EMPTY_SUMMARY, encoding="utf-8")
        if not memory.is_file():
            memory.write_text(_EMPTY_MEMORY, encoding="utf-8")
        if not raw.is_file():
            raw.write_text("# raw_memories\n\n", encoding="utf-8")

    # -- 读 -----------------------------------------------------------------

    def resolve_rel(self, rel: str) -> Path:
        """解析相对 memories 根的路径；拒绝越界与绝对路径。"""
        rel = rel.strip().replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise PermissionError(f"invalid memory path: {rel!r}")
        path = (self.root / rel).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as exc:
            raise PermissionError(f"path outside memories: {rel}") from exc
        return path

    def read_rel(self, rel: str, *, limit: int | None = READ_LIMIT) -> str:
        """读相对路径文件；limit=None 表示全文（consolidation 必须全文）。"""
        self.ensure_layout()
        path = self.resolve_rel(rel)
        if not path.is_file():
            raise FileNotFoundError(rel)
        text = path.read_text(encoding="utf-8", errors="replace")
        if limit is None:
            return text
        return text[:limit]

    def read_summary(self, *, limit: int | None = SUMMARY_INJECT_LIMIT) -> str:
        self.ensure_layout()
        try:
            text = (self.root / SUMMARY_NAME).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        if limit is None:
            return text
        return text[:limit]

    def summary_is_placeholder(self) -> bool:
        """summary 是否仍是空模板（stage2 尚未合并出有效摘要）。"""
        text = self.read_summary(limit=None)
        return not text.strip() or text.strip() == _EMPTY_SUMMARY.strip()

    def memory_has_entries(self) -> bool:
        """MEMORY.md 是否已有超过空模板的内容。"""
        try:
            text = self.read_rel(MEMORY_NAME, limit=None)
        except (FileNotFoundError, OSError, PermissionError):
            return False
        body = text.strip()
        return bool(body) and body != _EMPTY_MEMORY.strip()

    # -- 写 -----------------------------------------------------------------

    def append_raw(self, body: str) -> None:
        """stage1 流水追加（raw_memories.md）。"""
        self.ensure_layout()
        path = self.root / RAW_NAME
        with path.open("a", encoding="utf-8") as fh:
            fh.write(body)
            if not body.endswith("\n"):
                fh.write("\n")

    def write_rollout(self, session_id: str, body: str) -> str:
        """写入单轮摘要文件，返回相对路径（MEMORY 可引用）。"""
        self.ensure_layout()
        safe_sid = re.sub(r"[^a-zA-Z0-9_-]+", "-", session_id)[:40] or "sess"
        # 短随机后缀（评审 F5）：时间戳秒级精度，同 session 同秒两批会静默覆盖
        uniq = uuid.uuid4().hex[:6]
        rel = f"{ROLLOUTS_DIR}/{safe_sid}-{_utc_stamp()}-{uniq}.md"
        path = self.resolve_rel(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body.strip() + "\n", encoding="utf-8")
        return rel

    def atomic_write(self, rel: str, body: str) -> None:
        """原子覆盖（tmp + rename）；用于 MEMORY.md / memory_summary.md。"""
        self.ensure_layout()
        path = self.resolve_rel(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8")
        tmp.replace(path)

    # -- 检索与清理 -----------------------------------------------------------

    def grep(self, query: str, *, limit: int = 30) -> str:
        """在 MEMORY.md 与 rollout_summaries 中做子串搜索（空格分隔 AND）。"""
        self.ensure_layout()
        words = [w.lower() for w in query.split() if w]
        if not words:
            return "(empty query)"
        hits: list[str] = []
        files = [self.root / MEMORY_NAME]
        rollouts = self.root / ROLLOUTS_DIR
        if rollouts.is_dir():
            files.extend(sorted(rollouts.glob("*.md"), reverse=True)[:50])
        for path in files:
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            rel = path.relative_to(self.root).as_posix()
            for i, line in enumerate(lines, start=1):
                lower = line.lower()
                if all(w in lower for w in words):
                    hits.append(f"{rel}:{i}: {line.strip()}")
                    if len(hits) >= limit:
                        return "\n".join(hits)
        return "\n".join(hits) if hits else "(no matches)"

    def clear(self) -> None:
        """清空记忆目录并重建空模板（/memory clear；不动 episodic.jsonl）。"""
        if self.root.is_dir():
            for child in self.root.rglob("*"):
                if child.is_file():
                    child.unlink(missing_ok=True)
            for child in sorted(self.root.rglob("*"), reverse=True):
                if child.is_dir():
                    try:
                        child.rmdir()
                    except OSError:
                        pass
        self.ensure_layout()


def summary_prompt_block(store: MemoryFileStore) -> str:
    """组装塞进 system 的长期记忆段落（只注入 summary 截断 + 读指引）。

    故意不注入 MEMORY 全文，避免每轮烧大量 token；模型若需要细节，
    用 memory_read / memory_grep 工具按需读（claude/codex 检索思路）。
    """
    store.ensure_layout()
    summary = store.read_summary()
    if not summary.strip() or summary.strip() == _EMPTY_SUMMARY.strip():
        body = "（尚无浓缩摘要；需要时 memory_read MEMORY.md）"
    else:
        body = summary
    root = str(store.root)
    return (
        "## 长期记忆（系统生成，勿当项目规范）\n"
        f"记忆目录（memory_read / memory_grep 的根）：`{root}`\n"
        f"- 下方已提供 `{SUMMARY_NAME}`，**不要**再 memory_read 它。\n"
        f"- 需要细节：先 `memory_grep` 或 `memory_read(\"{MEMORY_NAME}\")`；\n"
        f"  仅当 MEMORY 指向具体 rollout 时再读 `rollout_summaries/...`（最多 1～2 个）。\n"
        f"\n### {SUMMARY_NAME}\n{body}"
    )


def format_raw_append(
    *,
    session_id: str,
    bullets: list[str],
    rollout_rel: str | None,
) -> str:
    """stage1 追加到 raw_memories.md 的流水块。"""
    lines = [
        "\n---\n",
        f"## {_utc_now()}  session={session_id}\n",
    ]
    if bullets:
        lines.append("### bullets\n")
        for b in bullets:
            lines.append(f"- {b}\n")
    if rollout_rel:
        lines.append(f"\nrollout: {rollout_rel}\n")
    return "".join(lines)
