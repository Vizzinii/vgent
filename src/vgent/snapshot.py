"""M12-B 快照/恢复（HANDOFF P9）：write/edit 改盘前登记 + 每回合版本化快照。

思路来源（只学思路不抄代码）：
- claude fileHistory（src/utils/fileHistory.ts）：工具写盘前 track 原文、每回合末统一
  版本化（同一文件跨回合多版本）、清单随会话持久化、30 天 GC；
- xcode-py SnapshotStore（src/runtime/snapshot.py）：sha256 去重 blob、last/named/undo
  三档恢复、open_turn 崩溃提升、blob GC。
vgent 适配：sync、工具 handler 契约不动（登记放 agent._dispatch_tool 派发层）、
路径按工具同规则（相对 os.getcwd()）。

布局（本机 ~/.vgent/checkpoints/<session_id>/，不进同步盘，决策 7）：
  snapshots.json     回合快照清单（保留最近 MAX_SNAPSHOTS 条）
  open_turn.json     本轮进行中的登记；崩溃/未封存时构造时提升为快照
  pre_restore.json   上一次 restore 前的状态（/restore undo 用）
  session_files.json 本会话碰过的文件清单（/snapshot 拍全量用）
  named/<slug>.json  命名档（最多 MAX_NAMED 个，超限淘汰最旧）
  blobs/<sha256>     内容去重的原文备份（引用计数 GC）

语义：回合 N 的快照 = 该回合首次 write/edit 写盘前的文件状态（= 上一回合结束后的状态）。
restore 到回合 N = 文件回到该回合开始前；对话历史不动（claude rewind 语义）。
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

MAX_SNAPSHOTS = 20  # 回合快照保留上限（claude fileHistory MAX_SNAPSHOTS=20）
MAX_NAMED = 20  # 命名档上限（xcode-py）
MAX_FILE_BYTES = 5 * 1024 * 1024  # 超大文件不备份（skip）
CLEANUP_DEFAULT_DAYS = 30  # 超期会话快照目录清理（claude cleanup 思路）

_OPEN = "open_turn.json"
_SNAPSHOTS = "snapshots.json"
_PRE = "pre_restore.json"
_SESSION_FILES = "session_files.json"
_NAMED = "named"
_BLOBS = "blobs"

_RESERVED_NAMES = frozenset({"last", "undo", "pre_restore"})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """临时文件 + rename 原子写，避免写半截留下坏 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def normalize_snapshot_name(raw: str | None) -> str:
    """空名 → UTC 时间戳；拒绝保留字与路径分隔符。"""
    name = (raw or "").strip()
    if not name:
        return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if name.lower() in _RESERVED_NAMES:
        raise ValueError(f"快照名 {name!r} 是保留字（last/undo/pre_restore）")
    if any(sep in name for sep in ("/", "\\", "\0")):
        raise ValueError("快照名不能包含路径分隔符")
    if name in {".", ".."}:
        raise ValueError("非法快照名")
    return name[:80]


def _safe_named_path(named_dir: Path, name: str) -> Path:
    slug = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE).strip("._") or "snap"
    return named_dir / f"{slug}.json"


@dataclass
class RestoreReport:
    """一次 restore 的结果摘要（CLI/Web 展示用）。"""

    restored: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def format(self) -> str:
        lines: list[str] = []
        if self.restored:
            lines.append("已恢复：" + ", ".join(self.restored))
        if self.deleted:
            lines.append("已删除：" + ", ".join(self.deleted))
        if self.skipped:
            bits = [f"{p}（{why}）" for p, why in self.skipped]
            lines.append("跳过：" + ", ".join(bits))
        if not lines:
            return "没有可恢复的内容"
        return "\n".join(lines)


class SnapshotStore:
    """绑定一个会话目录的快照存取（只跟踪工作区内文件，跨回合多版本）。"""

    def __init__(self, session_dir: Path, cwd: Path) -> None:
        self.session_dir = session_dir
        self.cwd = cwd.resolve()
        self.root = session_dir / "snapshots"
        self._open_files: dict[str, dict[str, Any]] = {}  # 本回合登记 {rel: entry}
        self._backed: set[str] = set()  # 本回合已登记路径（幂等）
        self._adopt_crashed_open()

    # -- 回合生命周期 -------------------------------------------------------

    def _adopt_crashed_open(self) -> None:
        """构造时：残留未封存的 open_turn 提升为回合快照（崩溃后 /restore 可用）。"""
        leftover = _read_json(self.root / _OPEN)
        if leftover and leftover.get("files"):
            self._append_snapshot(leftover.get("files", {}))

    def begin_turn(self) -> None:
        """新用户回合：清空本轮登记（run_turn 开头调用）。"""
        self._open_files = {}
        self._backed.clear()
        self._save_open()

    def seal_turn(self) -> None:
        """回合结束：有登记才追加一条回合快照（run_turn 的 finally 调用）。"""
        if self._open_files:
            self._append_snapshot(self._open_files)
        self._open_files = {}
        self._backed.clear()
        self._save_open()
        self._gc_blobs()

    def note_before_write(self, rel: str) -> None:
        """改盘前登记：本回合同文件只登记一次（写盘前原文拷入 blob）。"""
        rel = rel.strip().replace("\\", "/")
        if not rel or rel in self._backed:
            return
        entry = self._capture(rel)
        self._open_files[rel] = entry
        self._backed.add(rel)
        self._save_open()
        self._remember_path(rel)

    # -- 命名档 -------------------------------------------------------------

    def save_named(self, name: str | None) -> str:
        """按 session_files 拍此刻全部内容，写入命名档；返回最终名字。"""
        final = normalize_snapshot_name(name)
        files = self.capture_session_now()
        _atomic_write_json(
            _safe_named_path(self.root / _NAMED, final),
            {"v": 1, "kind": "named", "name": final, "ts": _utc_now(), "files": files},
        )
        self._evict_named()
        self._gc_blobs()
        return final

    def capture_session_now(self) -> dict[str, dict[str, Any]]:
        """本会话登记过的文件此刻的内容切片。"""
        files: dict[str, dict[str, Any]] = {}
        for rel in self._session_paths():
            files[rel] = self._capture(rel)
        return files

    # -- 恢复 ---------------------------------------------------------------

    def restore_last(self) -> RestoreReport:
        items = self._snapshot_items()
        if not items:
            return RestoreReport()
        return self._restore(items[-1].get("files") or {})

    def restore_index(self, index: int) -> RestoreReport:
        """按 list_entries 编号恢复（1 = 最新回合快照）。"""
        items = self._snapshot_items()
        if not items:
            return RestoreReport()
        if index < 1 or index > len(items):
            raise ValueError(f"快照编号超出范围（1..{len(items)}）")
        return self._restore(items[len(items) - index].get("files") or {})

    def restore_named(self, name: str) -> RestoreReport:
        path = self._find_named(name)
        if path is None:
            raise FileNotFoundError(f"命名快照不存在：{name}")
        data = _read_json(path) or {}
        return self._restore(data.get("files") or {})

    def restore_undo(self) -> RestoreReport:
        """回到上一次 restore 之前（pre_restore 栈，只弹一次）。"""
        data = _read_json(self.root / _PRE)
        files = (data or {}).get("files") or {}
        if not files:
            return RestoreReport()
        return self._restore(files, save_pre=False)

    def _restore(self, files: dict[str, Any], *, save_pre: bool = True) -> RestoreReport:
        """恢复：missing → 删当前文件；sha256 → 从 blob 写回；skip → 跳过。"""
        if save_pre:
            pre_files = self.capture_session_now()
            for rel in files:
                if rel not in pre_files:
                    pre_files[rel] = self._capture(rel)
            _atomic_write_json(
                self.root / _PRE,
                {"v": 1, "kind": "pre", "ts": _utc_now(), "files": pre_files},
            )
        report = RestoreReport()
        for rel, entry in files.items():
            if not isinstance(entry, dict):
                report.skipped.append((rel, "bad entry"))
                continue
            if entry.get("skip"):
                report.skipped.append((rel, str(entry.get("skip"))))
                continue
            path = self._resolve(rel)
            if path is None:
                report.skipped.append((rel, "out of workspace"))
                continue
            if entry.get("missing"):
                if path.is_file():
                    try:
                        path.unlink()
                        report.deleted.append(rel)
                    except OSError as exc:
                        report.skipped.append((rel, str(exc)))
                continue
            digest = entry.get("sha256")
            if not isinstance(digest, str) or not digest:
                report.skipped.append((rel, "no blob"))
                continue
            blob = self.root / _BLOBS / digest
            if not blob.is_file():
                report.skipped.append((rel, "blob missing"))
                continue
            try:
                data = blob.read_bytes()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                report.restored.append(rel)
            except OSError as exc:
                report.skipped.append((rel, str(exc)))
        return report

    # -- 展示 ---------------------------------------------------------------

    def list_entries(self) -> list[dict[str, Any]]:
        """列出可恢复项：回合快照（编号 1=最新）、命名档、undo。"""
        rows: list[dict[str, Any]] = []
        items = self._snapshot_items()
        n = len(items)
        for i, rec in enumerate(reversed(items)):
            rows.append(
                {
                    "key": str(n - i),
                    "kind": "turn",
                    "label": f"回合 {n - i}",
                    "ts": str(rec.get("ts") or ""),
                    "files": sorted((rec.get("files") or {}).keys()),
                }
            )
        named_dir = self.root / _NAMED
        if named_dir.is_dir():
            for child in sorted(named_dir.glob("*.json")):
                data = _read_json(child) or {}
                rows.append(
                    {
                        "key": str(data.get("name") or child.stem),
                        "kind": "named",
                        "label": "命名档",
                        "ts": str(data.get("ts") or ""),
                        "files": sorted((data.get("files") or {}).keys()),
                    }
                )
        pre = _read_json(self.root / _PRE)
        if pre and pre.get("files"):
            rows.append(
                {
                    "key": "undo",
                    "kind": "undo",
                    "label": "撤回前",
                    "ts": str(pre.get("ts") or ""),
                    "files": sorted((pre.get("files") or {}).keys()),
                }
            )
        return rows

    def snapshot_count(self) -> int:
        return len(self._snapshot_items())

    # -- 内部 ---------------------------------------------------------------

    def _capture(self, rel: str) -> dict[str, Any]:
        """登记单文件：拷原文入 blob（sha256 去重）；missing/skip 打标。"""
        path = self._resolve(rel)
        version = self._next_version(rel)
        if path is None:
            return {"skip": "out of workspace", "version": version}
        if not path.exists():
            return {"missing": True, "version": version}
        if path.is_dir():
            return {"skip": "is_directory", "version": version}
        try:
            size = path.stat().st_size
        except OSError as exc:
            return {"skip": str(exc), "version": version}
        if size > MAX_FILE_BYTES:
            return {"skip": "too_large", "version": version}
        try:
            data = path.read_bytes()
        except OSError as exc:
            return {"skip": str(exc), "version": version}
        digest = sha256(data).hexdigest()
        blob = self.root / _BLOBS / digest
        if not blob.is_file():
            blob.parent.mkdir(parents=True, exist_ok=True)
            tmp = blob.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(blob)
        return {"sha256": digest, "version": version}

    def _resolve(self, rel: str) -> Path | None:
        """相对 cwd 解析；绝对路径或越界（..）返回 None。"""
        p = Path(rel)
        if p.is_absolute():
            return None
        resolved = (self.cwd / p).resolve(strict=False)
        try:
            resolved.relative_to(self.cwd)
        except ValueError:
            return None
        return resolved

    def _next_version(self, rel: str) -> int:
        """该文件跨回合的备份序号（claude @v{n} 思路；展示用）。"""
        best = 0
        for files in self._all_record_files():
            entry = files.get(rel)
            if isinstance(entry, dict):
                best = max(best, int(entry.get("version") or 0))
        return best + 1

    def _append_snapshot(self, files: dict[str, Any]) -> None:
        data = _read_json(self.root / _SNAPSHOTS) or {"v": 1, "items": []}
        items = data.get("items") or []
        items = [it for it in items if isinstance(it, dict)]
        items.append({"ts": _utc_now(), "files": files})
        items = items[-MAX_SNAPSHOTS:]
        _atomic_write_json(self.root / _SNAPSHOTS, {"v": 1, "items": items})

    def _snapshot_items(self) -> list[dict[str, Any]]:
        data = _read_json(self.root / _SNAPSHOTS) or {}
        items = data.get("items") or []
        return [it for it in items if isinstance(it, dict)]

    def _save_open(self) -> None:
        _atomic_write_json(
            self.root / _OPEN,
            {"v": 1, "kind": "open", "ts": _utc_now(), "files": self._open_files},
        )

    def _session_paths(self) -> list[str]:
        data = _read_json(self.root / _SESSION_FILES) or {}
        raw = data.get("paths") or []
        return [str(p) for p in raw if isinstance(p, str) and p]

    def _remember_path(self, rel: str) -> None:
        paths = self._session_paths()
        if rel not in paths:
            paths.append(rel)
            _atomic_write_json(self.root / _SESSION_FILES, {"v": 1, "paths": paths})

    def _find_named(self, name: str) -> Path | None:
        named_dir = self.root / _NAMED
        if not named_dir.is_dir():
            return None
        exact = _safe_named_path(named_dir, name)
        if exact.is_file():
            return exact
        for child in named_dir.glob("*.json"):
            data = _read_json(child) or {}
            if data.get("name") == name:
                return child
        return None

    def _evict_named(self) -> None:
        named_dir = self.root / _NAMED
        if not named_dir.is_dir():
            return
        items: list[tuple[str, Path]] = []
        for child in named_dir.glob("*.json"):
            data = _read_json(child) or {}
            items.append((str(data.get("ts") or ""), child))
        if len(items) <= MAX_NAMED:
            return
        items.sort(key=lambda it: it[0])
        for _, path in items[: len(items) - MAX_NAMED]:
            try:
                path.unlink()
            except OSError:
                pass

    def _all_record_files(self) -> list[dict[str, Any]]:
        """全部记录（回合快照 + open + pre + named）里的 files 字典（blob GC 用）。"""
        out: list[dict[str, Any]] = []
        for rec in self._snapshot_items():
            files = rec.get("files")
            if isinstance(files, dict):
                out.append(files)
        for path in (self.root / _OPEN, self.root / _PRE):
            data = _read_json(path)
            if data and isinstance(data.get("files"), dict):
                out.append(data["files"])
        named_dir = self.root / _NAMED
        if named_dir.is_dir():
            for child in named_dir.glob("*.json"):
                data = _read_json(child) or {}
                if isinstance(data.get("files"), dict):
                    out.append(data["files"])
        return out

    def _gc_blobs(self) -> None:
        """删除无任何记录引用的 blob（引用计数 GC）。"""
        blob_dir = self.root / _BLOBS
        if not blob_dir.is_dir():
            return
        live: set[str] = set()
        for files in self._all_record_files():
            for entry in files.values():
                if isinstance(entry, dict):
                    digest = entry.get("sha256")
                    if isinstance(digest, str) and digest:
                        live.add(digest)
        for blob in blob_dir.iterdir():
            if blob.is_file() and blob.name not in live:
                try:
                    blob.unlink()
                except OSError:
                    pass

    @staticmethod
    def cleanup_old(checkpoints_root: Path, days: int = CLEANUP_DEFAULT_DAYS) -> int:
        """删除 checkpoints 下超期未活动的会话快照目录；返回删除数。"""
        if not checkpoints_root.is_dir():
            return 0
        cutoff = time.time() - days * 86400
        removed = 0
        for child in checkpoints_root.iterdir():
            if not child.is_dir():
                continue
            try:
                mtime = child.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                try:
                    shutil.rmtree(child)
                    removed += 1
                except OSError:
                    pass
        return removed
