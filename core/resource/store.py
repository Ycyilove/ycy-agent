"""统一资源服务：所有来源（工具 / RAG / 附件 / 对话）的二进制资源都走这里。

存储：data/resources/<id>/<filename> + meta.json
索引：data/resources/index.jsonl（追加写）
URL：/api/resources/<id>

安全：
    - MIME 白名单 + 扩展名黑名单 + 大小限制
    - HTML / SVG 拒绝（防 XSS）
    - 24 小时 TTL
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── 配置 ──

_ROOT = Path(os.getenv(
    "AGENT_RESOURCE_DIR",
    str(Path(__file__).resolve().parents[2] / "data" / "resources"),
)).resolve()
_MAX_FILE_BYTES = int(os.getenv("AGENT_RESOURCE_MAX_FILE_MB", "50")) * 1024 * 1024
_TTL_SECONDS = int(os.getenv("AGENT_RESOURCE_TTL_HOURS", "24")) * 3600


# ── 安全过滤 ──

_ALLOWED_MIME_PREFIXES = ("image/", "text/", "audio/", "video/")

_ALLOWED_MIME_EXACT = {
    "application/pdf",
    "application/json",
    "application/xml",
    "application/zip",
    "application/x-tar",
    "application/gzip",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/octet-stream",   # 兜底，允许但按扩展名再判
}

_BLOCKED_EXT = {
    ".exe", ".com", ".scr", ".pif", ".bat", ".cmd", ".msi", ".msp",
    ".dll", ".so", ".dylib", ".run", ".jar",
    ".sh", ".bash", ".zsh", ".ps1", ".psm1",
    ".vbs", ".vbe", ".jse", ".wsf", ".wsh",
    ".lnk", ".url", ".reg",
    ".app", ".deb", ".rpm", ".dmg", ".pkg",
}

_BLOCKED_MIME = {
    "application/x-msdownload",
    "application/x-msdos-program",
    "application/x-executable",
    "application/x-sharedlib",
    "application/x-sh",
    "application/x-shellscript",
    "application/x-bat",
    "application/vnd.microsoft.portable-executable",
    "application/java-archive",
    "application/x-httpd-php",
    "text/html",
    "image/svg+xml",
    "application/xhtml+xml",
}


def _safe_filename(name: str) -> str:
    name = (name or "resource").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w\u4e00-\u9fff.\-]+", "_", name)
    return name[:150] or "resource"


_KIND_RULES: List[tuple] = [
    (lambda m, e: m.startswith("image/"), "image"),
    (lambda m, e: m == "application/pdf" or e == ".pdf", "pdf"),
    (lambda m, e: "spreadsheet" in m or e in (".xlsx", ".xls", ".ods"), "excel"),
    (lambda m, e: "wordprocessing" in m or e in (".docx", ".doc", ".odt"), "word"),
    (lambda m, e: "presentation" in m or e in (".pptx", ".ppt", ".odp"), "ppt"),
    (lambda m, e: m.startswith("audio/"), "audio"),
    (lambda m, e: m.startswith("video/"), "video"),
    (lambda m, e: m == "application/zip" or e in (".zip", ".tar", ".gz", ".7z", ".rar"), "archive"),
    (lambda m, e: m.startswith("text/") or e in (".txt", ".md", ".log", ".csv", ".json"), "text"),
]


def _guess_kind(mime: str, ext: str) -> str:
    m = (mime or "").lower()
    e = (ext or "").lower()
    for rule, kind in _KIND_RULES:
        try:
            if rule(m, e):
                return kind
        except Exception:
            continue
    return "file"


def _is_allowed(mime: str, ext: str) -> tuple:
    m = (mime or "").lower().strip()
    e = (ext or "").lower().strip()
    if e in _BLOCKED_EXT:
        return False, f"扩展名被拒绝: {e}"
    if m in _BLOCKED_MIME:
        return False, f"MIME 被拒绝: {m}"
    if m in _ALLOWED_MIME_EXACT:
        return True, ""
    for prefix in _ALLOWED_MIME_PREFIXES:
        if m.startswith(prefix):
            return True, ""
    return False, f"不支持的 MIME: {m or '(空)'}"


# ── 数据模型 ──

@dataclass
class UnifiedResource:
    id: str
    kind: str
    filename: str
    mime: str
    size: int
    url: str
    source: str = "tool"           # tool / rag / attachment / llm
    source_tool: str = ""
    task_id: str = ""
    session_id: str = ""
    created_at: float = field(default_factory=time.time)
    page_no: Optional[int] = None
    columns: Optional[List[str]] = None
    rows: Optional[List[Dict[str, Any]]] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "kind": self.kind,
            "filename": self.filename,
            "mime": self.mime,
            "size": self.size,
            "url": self.url,
            "source": self.source,
            "source_tool": self.source_tool,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
        }
        if self.page_no is not None:
            d["page_no"] = self.page_no
        if self.columns is not None:
            d["columns"] = self.columns
        if self.rows is not None:
            d["rows"] = self.rows
        if self.extra:
            d["extra"] = self.extra
        return d


# ── 服务 ──

_lock = threading.RLock()
_cache: Dict[str, UnifiedResource] = {}
_index_loaded = False


def _ensure_root():
    _ROOT.mkdir(parents=True, exist_ok=True)


def _meta_path(rid: str) -> Path:
    return _ROOT / rid / "meta.json"


def _file_dir(rid: str) -> Path:
    return _ROOT / rid


def _index_path() -> Path:
    return _ROOT / "index.jsonl"


def _load_index():
    global _cache, _index_loaded
    if _index_loaded:
        return
    with _lock:
        if _index_loaded:
            return
        _cache = {}
        _ensure_root()
        p = _index_path()
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            rid = data.get("id")
                            if rid:
                                _cache[rid] = UnifiedResource(**{
                                    k: v for k, v in data.items()
                                    if k in UnifiedResource.__dataclass_fields__
                                })
                        except Exception:
                            continue
            except Exception:
                logger.exception("[resource] 索引加载失败")
        _index_loaded = True


def _append_index(res: UnifiedResource):
    _ensure_root()
    line = json.dumps(res.to_dict(), ensure_ascii=False) + "\n"
    with _lock:
        with open(_index_path(), "a", encoding="utf-8") as f:
            f.write(line)
        _cache[res.id] = res


def _write_meta(res: UnifiedResource):
    d = _file_dir(res.id)
    d.mkdir(parents=True, exist_ok=True)
    tmp = _meta_path(res.id).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(tmp, _meta_path(res.id))


def _persist(res: UnifiedResource):
    _write_meta(res)
    _append_index(res)


# ── 对外 API ──

def save_bytes(
    content: bytes,
    *,
    filename: str,
    mime: str = "",
    source: str = "tool",
    source_tool: str = "",
    task_id: str = "",
    session_id: str = "",
    kind_override: Optional[str] = None,
) -> Optional[UnifiedResource]:
    if not content:
        return None
    if len(content) > _MAX_FILE_BYTES:
        logger.warning("[resource] 文件超过限制: %s (%d > %d)",
                       filename, len(content), _MAX_FILE_BYTES)
        return None

    safe_name = _safe_filename(filename)
    ext = Path(safe_name).suffix
    if not mime:
        mime = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"

    allowed, reason = _is_allowed(mime, ext)
    if not allowed:
        logger.warning("[resource] 拒绝 %s (mime=%s): %s", safe_name, mime, reason)
        return None

    rid = uuid.uuid4().hex
    with _lock:
        _ensure_root()
        d = _file_dir(rid)
        d.mkdir(parents=True, exist_ok=True)
        try:
            (d / safe_name).write_bytes(content)
        except Exception:
            logger.exception("[resource] 写文件失败")
            shutil.rmtree(d, ignore_errors=True)
            return None

        res = UnifiedResource(
            id=rid,
            kind=kind_override or _guess_kind(mime, ext),
            filename=safe_name,
            mime=mime,
            size=len(content),
            url=f"/api/resources/{rid}",
            source=source,
            source_tool=source_tool,
            task_id=task_id,
            session_id=session_id,
        )
        _persist(res)
        logger.info("[resource] saved %s (%s, %d bytes) tool=%s",
                    rid, res.kind, len(content), source_tool)
        return res


def save_base64(b64: str, **kwargs) -> Optional[UnifiedResource]:
    try:
        if b64.startswith("data:") and "," in b64:
            header, _, b64 = b64.partition(",")
            if not kwargs.get("mime") and ";" in header:
                kwargs["mime"] = header[5:].split(";")[0]
        raw = base64.b64decode(b64, validate=False)
    except Exception:
        logger.exception("[resource] base64 解码失败")
        return None
    return save_bytes(raw, **kwargs)


def save_file(
    src_path: str,
    *,
    filename: str = "",
    mime: str = "",
    source: str = "tool",
    source_tool: str = "",
    task_id: str = "",
    session_id: str = "",
) -> Optional[UnifiedResource]:
    p = Path(src_path)
    if not p.is_file():
        return None
    try:
        size = p.stat().st_size
    except OSError:
        return None
    if size > _MAX_FILE_BYTES:
        logger.warning("[resource] 文件过大: %s (%d bytes)", src_path, size)
        return None

    name = filename or p.name
    safe_name = _safe_filename(name)
    ext = Path(safe_name).suffix
    if not mime:
        mime = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    allowed, reason = _is_allowed(mime, ext)
    if not allowed:
        logger.warning("[resource] 拒绝 %s: %s", safe_name, reason)
        return None

    rid = uuid.uuid4().hex
    with _lock:
        _ensure_root()
        d = _file_dir(rid)
        d.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(p, d / safe_name)
        except Exception:
            logger.exception("[resource] 复制失败: %s", src_path)
            shutil.rmtree(d, ignore_errors=True)
            return None

        res = UnifiedResource(
            id=rid,
            kind=_guess_kind(mime, ext),
            filename=safe_name,
            mime=mime,
            size=size,
            url=f"/api/resources/{rid}",
            source=source,
            source_tool=source_tool,
            task_id=task_id,
            session_id=session_id,
        )
        _persist(res)
        return res


def save_text(
    text: str,
    *,
    filename: str = "output.txt",
    source: str = "tool",
    source_tool: str = "",
    task_id: str = "",
    session_id: str = "",
) -> Optional[UnifiedResource]:
    return save_bytes(
        text.encode("utf-8"),
        filename=filename,
        mime="text/plain",
        source=source,
        source_tool=source_tool,
        task_id=task_id,
        session_id=session_id,
    )


def save_table(
    columns: List[str],
    rows: List[Dict[str, Any]],
    *,
    filename: str = "table",
    source: str = "tool",
    source_tool: str = "",
    task_id: str = "",
    session_id: str = "",
) -> UnifiedResource:
    rid = uuid.uuid4().hex
    res = UnifiedResource(
        id=rid,
        kind="table",
        filename=filename,
        mime="application/x-table",
        size=len(rows),
        url=f"/api/resources/{rid}",
        source=source,
        source_tool=source_tool,
        task_id=task_id,
        session_id=session_id,
        columns=columns,
        rows=rows,
    )
    with _lock:
        _persist(res)
    return res


def get(resource_id: str) -> Optional[UnifiedResource]:
    _load_index()
    with _lock:
        return _cache.get(resource_id)


def get_path(resource_id: str) -> Optional[Path]:
    res = get(resource_id)
    if res is None or res.kind == "table":
        return None
    p = _file_dir(resource_id) / res.filename
    return p if p.exists() else None


def delete(resource_id: str) -> bool:
    _load_index()
    with _lock:
        res = _cache.pop(resource_id, None)
        if res is None:
            return False
        shutil.rmtree(_file_dir(resource_id), ignore_errors=True)
        return True


def cleanup_expired() -> int:
    if not _ROOT.exists():
        return 0
    _load_index()
    now = time.time()
    count = 0
    with _lock:
        expired = [rid for rid, r in _cache.items()
                   if now - r.created_at > _TTL_SECONDS]
        for rid in expired:
            shutil.rmtree(_file_dir(rid), ignore_errors=True)
            _cache.pop(rid, None)
            count += 1
        if count:
            # 重建索引
            try:
                tmp = _index_path().with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    for r in _cache.values():
                        f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
                os.replace(tmp, _index_path())
            except Exception:
                logger.exception("[resource] 索引重建失败")
            logger.info("[resource] 清理了 %d 个过期资源", count)
    return count


def stats() -> Dict[str, Any]:
    _load_index()
    with _lock:
        return {"total": len(_cache), "dir": str(_ROOT)}