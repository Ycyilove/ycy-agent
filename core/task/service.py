"""Task orchestration for the Agent workbench.

关键设计：
    - RAG 兜底已移除——让 orchestrator 通过 search_knowledge 工具自己决策
    - thinking/answer 流解析抽到 stream_parser
    - prompt 构造抽到 prompts
    - 保留了 SSE 心跳、审批流、持久化等机制
"""

from __future__ import annotations

import asyncio
import inspect
import base64
import json
import logging
import mimetypes
import os
import threading
import time
import uuid
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    # 仅类型检查器可见；运行期不 import，不触发 deps 违规
    from tools.types import ToolCallRequest, ToolCallResult

from core.task.state import OrchestratorState
from core.task.modes import MODE_ASK, MODE_AGENT, MODES, validate_mode

from core.llm.prompts import (
    attachment_context as build_attachment_context,
    attachment_label,
    build_prompt,
)
from .stream_parser import (
    StreamSplitter,
    async_iter_sync as _async_iter_sync,
    parse_provider_chunk,
)

from infra.debug_log.llm import begin_turn, end_turn
from infra.observability import set_turn_context
from core.task.tool_result_store import build_handle_summary

logger = logging.getLogger(__name__)


@dataclass
class PendingApproval:
    task_id: str
    session_id: str
    step_id: str
    approval_id: str
    tool_call: Any
    title: str
    turn_id: str = ""
    path: Optional[str] = None

    # ── orchestrator 恢复所需信息 ──
    goal: str = ""
    mode: str = MODE_AGENT
    workspace: Optional[List[str]] = None
    workspace_trusted: bool = False
    auto_discover_mcp: bool = False
    step_index: int = 1
    state_snapshot: Optional[dict] = None

    # ── 内部：创建时间，用于 TTL 清理 ──
    _created_at: float = field(default_factory=time.time)

def _looks_base64(s: str) -> bool:
    if not isinstance(s, str) or len(s) < 32:
        return False
    return bool(re.match(r"^[A-Za-z0-9+/=\s]+$", s[:256]))


def _dedup_resources(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in items:
        rid = r.get("id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append(r)
    return out
    
def sse_event(event_type: str, **payload: Any) -> str:
    """Encode one JSON event for the browser's EventSource parser."""
    return (
        f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"
    )


async def with_sse_done(
    events: AsyncIterator[str],
    is_disconnected: Optional[Callable[[], Any]] = None,
) -> AsyncIterator[str]:
    """Append the conventional SSE terminator, with periodic heartbeat.

    is_disconnected: 可选，返回 bool 或 awaitable[bool] 的回调。
                     客户端断开时停止泵出事件并取消上游生成器。
    """
    HEARTBEAT_INTERVAL = 15.0

    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _DONE = object()
    disconnected = False

    async def _check_disconnected() -> bool:
        if is_disconnected is None:
            return False
        try:
            result = is_disconnected()
            if inspect.isawaitable(result):
                result = await result
            return bool(result)
        except Exception:
            return False

    async def pump():
        nonlocal disconnected
        try:
            async for event in events:
                if await _check_disconnected():
                    disconnected = True
                    logger.info("[dsh][sse] client disconnected, stop pumping")
                    break
                await queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("[dsh][sse] upstream generator raised")
            try:
                await queue.put(sse_event(
                    "error",
                    message=f"任务异常：{type(exc).__name__}: {exc}",
                ))
            except Exception:
                pass
        finally:
            try:
                await queue.put(_DONE)
            except Exception:
                pass

    pump_task = asyncio.create_task(pump())

    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=HEARTBEAT_INTERVAL
                )
            except asyncio.TimeoutError:
                if await _check_disconnected():
                    logger.info("[dsh][sse] client disconnected at heartbeat")
                    disconnected = True
                    break
                yield ": keep-alive\n\n"
                continue
            if item is _DONE:
                break
            yield item
    finally:
        if not pump_task.done():
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass

    if disconnected:
        logger.info("[dsh][sse] stream ended due to client disconnect")
        return
    yield "data: [DONE]\n\n"

class AgentTaskService:
    """Coordinate task planning, tool execution, approvals, and persistence."""

    def __init__(
        self,
        agent_provider: Callable[[], Any],
        llm_provider: Callable[[], Any],
        multimodal_provider: Callable[[], Any],
        session_provider: Callable[[], Any],
        rag_provider: Optional[Callable[[], Any]] = None,
        orchestrator_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._agent_provider = agent_provider
        self._llm_provider = llm_provider
        self._multimodal_provider = multimodal_provider
        self._session_provider = session_provider
        self._rag_provider = rag_provider
        self._orchestrator_provider = orchestrator_provider
        self._pending: Dict[str, PendingApproval] = {}
        self._lock = threading.RLock()
        self._pending_ttl_seconds = 3600   # 1 小时未审批自动清理

    # ────────── 待审批管理 ──────────

    def _store_pending(self, pending: "PendingApproval") -> None:
        with self._lock:
            self._purge_expired_locked()
            self._pending[pending.approval_id] = pending
        logger.info(
            "[dsh][pending] stored approval %s: task=%s session=%s tool=%s",
            pending.approval_id, pending.task_id,
            pending.session_id, pending.tool_call.tool_name,
        )
        try:
            from infra.debug_log.llm import dump_llm_call
            dump_llm_call(
                kind="approval_store",
                prompt="",
                response=None,
                tool_name=pending.tool_call.tool_name,
                extra={
                    "approval_id": pending.approval_id,
                    "task_id": pending.task_id,
                    "session_id": pending.session_id,
                    "tool_name": pending.tool_call.tool_name,
                    "params": pending.tool_call.parameters,
                    "title": pending.title,
                    "step_index": pending.step_index,
                },
            )
        except Exception:
            logger.exception("[dsh][pending] dump approval_store failed")

        # ── 同步到服务端权威 ApprovalStore（供网关 verify 用） ──
        try:
            from tools.security.approval import get_approval_store
            store = get_approval_store()
            # 若 orchestrator 已 create 过（同 approval_id），跳过
            existing = store._records.get(pending.approval_id)  # 内部读，仅同步用
            if existing is None:
                store.create(
                    task_id=pending.task_id,
                    session_id=pending.session_id,
                    tool_name=pending.tool_call.tool_name,
                    params=pending.tool_call.parameters,
                )
        except Exception:
            logger.exception("同步 ApprovalStore 失败: %s", pending.approval_id)

    def _take_pending(
        self, approval_id: str, task_id: str
    ) -> Optional["PendingApproval"]:
        with self._lock:
            self._purge_expired_locked()
            pending = self._pending.pop(approval_id, None)
        if pending is None or pending.task_id != task_id:
            return None
        return pending

    def _purge_expired_locked(self) -> None:
        """清理超过 TTL 的 pending。必须在锁内调用。"""
        now = time.time()
        expired = [
            aid for aid, p in self._pending.items()
            if now - getattr(p, "_created_at", now) > self._pending_ttl_seconds
        ]
        for aid in expired:
            self._pending.pop(aid, None)
        if expired:
            logger.info(
                "[dsh][pending] purged %d expired approvals: %r",
                len(expired), expired,
            )

    # ────────── 持久化 ──────────

    def _persist_task(
        self,
        task_id: str,
        session_id: str,
        message: str,
        mode: str,
        workspace: Optional[List[str]],
        attachments: List[Dict[str, Any]],
    ) -> None:
        memory = self._session_provider()
        title = (message or "").strip().replace("\n", " ")[:40]
        if not title and attachments:
            title = f"处理 {len(attachments)} 个附件"
        if not title:
            title = "新任务"
        if not hasattr(memory, "create_task"):
            return
        memory.create_task({
            "task_id": task_id,
            "session_id": session_id,
            "title": title,
            "status": "running",
            "mode": mode,
            "workspace": workspace or [],
            "attachments": [
                {"name": attachment_label(a), "type": a.get("content_type")}
                for a in (attachments or [])
            ],
        })

    def _persist_step(self, task_id: str, step: Dict[str, Any]) -> None:
        memory = self._session_provider()
        if not hasattr(memory, "upsert_step"):
            return
        try:
            memory.upsert_step(task_id, step)
        except Exception:
            logger.exception("持久化步骤失败: %s", step.get("id"))

    def _persist_task_status(self, task_id: str, status: str) -> None:
        memory = self._session_provider()
        if not hasattr(memory, "update_task"):
            return
        try:
            memory.update_task(task_id, status=status)
        except Exception:
            logger.exception("持久化任务状态失败: %s", task_id)

    def _persist_file_log(self, session_id: str, entry: Dict[str, Any]) -> None:
        """持久化文件操作日志。read 操作不落盘。"""
        if not session_id or not entry:
            return
        action = entry.get("action")
        if action == "read":
            return   # 读操作太频繁，不持久化
        memory = self._session_provider()
        if not hasattr(memory, "add_file_log"):
            return
        try:
            memory.add_file_log(session_id, entry)
        except Exception:
            logger.exception("持久化 file_log 失败: %s", entry.get("id"))

    def _emit_file_log(
        self, session_id: str, entry: Dict[str, Any]
    ) -> str:
        """持久化 + 发 SSE。"""
        self._persist_file_log(session_id, entry)
        return sse_event("file_log", entry=entry)

    def _emit_step(self, task_id: str, step: Dict[str, Any]) -> str:
        self._persist_step(task_id, step)
        return sse_event("step", step=step)

    # ────────── 会话 ──────────

    def _ensure_session(self, session_id: Optional[str], message: str) -> str:
        memory = self._session_provider()
        name = message.strip().replace("\n", " ")[:40] or "Agent 任务"
        if session_id:
            return memory.ensure_session(session_id, name)
        return memory.create_session(name)

    # ────────── Step 构造 ──────────

    @staticmethod
    def _step(
        step_id: str,
        label: str,
        status: str,
        kind: str = "think",
        **extra: Any,
    ) -> Dict[str, Any]:
        return {
            "id": step_id,
            "label": label,
            "status": status,
            "kind": kind,
            **extra,
        }

    # ────────── 工作区 ──────────

    @staticmethod
    def _normalise_workspace(workspace: Optional[List[str]]) -> List[str]:
        return [
            os.path.abspath(path)
            for path in (workspace or [])
            if str(path).strip()
        ]

    # ────────── 附件 ──────────

    @staticmethod
    def _attachment_label(attachment: Dict[str, Any]) -> str:
        return attachment_label(attachment)

    def _build_multimodal_messages(
        self,
        message: str,
        attachments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for attachment in attachments:
            filename = attachment_label(attachment)
            content_type = (
                attachment.get("content_type")
                or mimetypes.guess_type(filename)[0]
            )
            content_type = content_type or "application/octet-stream"
            source_type = "file" if content_type == "application/pdf" else "image"
            content.append({
                "type": source_type,
                "source": {
                    "type": "base64",
                    "media_type": content_type,
                    "data": base64.b64encode(
                        attachment.get("content") or b""
                    ).decode("ascii"),
                },
            })
        content.append({"type": "text", "text": message or "请分析这些附件。"})
        return [{"role": "user", "content": content}]

    def _build_multimodal_message(
        self,
        message: str,
        attachments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return self._build_multimodal_messages(message, attachments)[0]["content"]

    @staticmethod
    def _is_multimodal(attachment: Dict[str, Any]) -> bool:
        filename = attachment_label(attachment)
        content_type = (attachment.get("content_type") or "").lower()
        suffix = Path(filename).suffix.lower()
        return content_type.startswith("image/") or suffix == ".pdf"

    # ────────── 会话输入记录 ──────────

    def _record_session_input(
        self,
        session_id: str,
        message: str,
        attachments: List[Dict[str, Any]],
    ) -> None:
        memory = self._session_provider()
        file_ids = []
        for attachment in attachments:
            file_ids.append(
                memory.add_session_file(
                    session_id,
                    attachment_label(attachment),
                    file_type=attachment.get("content_type"),
                )
            )
        memory.add_message(
            session_id,
            "user",
            message or "请处理附件。",
            file_ids=file_ids,
        )

    # ────────── 工具辅助 ──────────

    @staticmethod
    def _tool_path(tool_call: ToolCallRequest) -> Optional[str]:
        for key in ("file_path", "old_path", "directory", "path", "output_path"):
            value = tool_call.parameters.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    # 内置工具 → 动作映射
    _FILE_ACTION_MAP: Dict[str, str] = {
        # 读
        "list_files": "read",
        "read_csv": "read",
        "read_excel": "read",
        "get_file_info": "read",
        "read_text_file": "read",
        "read_file_range": "read",
        "read_pdf_text": "read",
        "read_pdf_metadata": "read",
        "read_docx_text": "read",
        "read_docx_tables": "read",
        "search_content": "read",
        "search_files": "read",
        # 写
        "write_file": "write",
        "edit_file": "write",
        "write_excel": "write",
        "edit_excel": "write",
        "add_excel_sheet": "write",
        "write_docx": "write",
        "edit_docx": "write",
        "convert_file_format": "write",
        "merge_pdfs": "write",
        "split_pdf": "write",
        "create_directory": "write",
        # 移动
        "rename_file": "move",
        "move_file": "move",
        # 删除（若将来新增）
        "delete_file": "delete",
        "remove_file": "delete",
    }

    @staticmethod
    def _infer_mcp_action(tool_name: str) -> Optional[str]:
        """从 MCP 工具名推断动作。"""
        if not tool_name.startswith("mcp__"):
            return None
        base = tool_name.rsplit("__", 1)[-1].lower()
        if any(k in base for k in ("write", "edit", "create", "append",
                                   "patch", "save", "update")):
            return "write"
        if any(k in base for k in ("move", "rename")):
            return "move"
        if any(k in base for k in ("delete", "remove", "unlink")):
            return "delete"
        if any(k in base for k in ("read", "list", "get", "search",
                                   "find", "stat", "info")):
            return "read"
        return None

    @staticmethod
    def _file_log_entry(
        tool_call: ToolCallRequest, result: ToolCallResult
    ) -> Optional[Dict[str, Any]]:
        action = AgentTaskService._FILE_ACTION_MAP.get(tool_call.tool_name)
        if not action:
            action = AgentTaskService._infer_mcp_action(tool_call.tool_name)
        if not action:
            return None

        path = AgentTaskService._tool_path(tool_call)
        if isinstance(result.result, dict):
            path = result.result.get("new_path") or path
        if not path:
            return None

        return {
            "id": f"log-{uuid.uuid4().hex}",
            "action": action,
            "path": path,
            "time": int(time.time() * 1000),
            "status": "success" if result.success else "failed",
        }

    def _collect_session_resources(self, session_id: str) -> List[Dict[str, Any]]:
        if not session_id:
            return []
        try:
            memory = self._session_provider()
            files = memory.get_session_files(session_id)
        except Exception:
            return []

        resources: List[Dict[str, Any]] = []
        for f in files:
            file_id = f.get("file_id")
            file_name = (f.get("file_name") or "").lower()
            file_type = (f.get("file_type") or "").lower()
            is_image = file_type.startswith("image/") or file_name.endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
            )
            if not is_image:
                continue
            resources.append({
                "id": f"session-{file_id}",
                "kind": "image",
                "filename": f.get("file_name") or "attachment",
                "url": f"/api/session/file/{file_id}/content",
                "source": "session",
            })
        return resources

    @staticmethod
    def _strip_bytes(obj: Any, _depth: int = 0) -> Any:
        """递归把 bytes 替换成占位字符串。

        与 json.dumps(default=str) 的分工：
        - _strip_bytes: bytes → "<binary N bytes>"（可读、可控长度）
        - default=str : 其他不可序列化对象（datetime、自定义类）→ str()
        两者**不能互相替代**：只留 default=str 会让大二进制变成
        几 MB 的 b'...'；只留 _strip_bytes 遇到 datetime 会抛。
        - dict / list / tuple 递归处理
        - 深度超限返回类型名，防止循环引用
        """
        if _depth > 10:
            return str(type(obj).__name__)
        if isinstance(obj, bytes):
            return f"<binary {len(obj)} bytes>"
        if isinstance(obj, dict):
            return {
                k: AgentTaskService._strip_bytes(v, _depth + 1)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [
                AgentTaskService._strip_bytes(v, _depth + 1)
                for v in obj
            ]
        if isinstance(obj, tuple):
            return tuple(
                AgentTaskService._strip_bytes(v, _depth + 1)
                for v in obj
            )
        return obj

    @staticmethod
    def _compact_tool_payload(payload: Any) -> str:
        """给 LLM 的工具返回摘要：优先 structuredContent，其次 content 文本。

        避免 MCP 的 content + structuredContent 双份同时进 prompt。
        """
        if isinstance(payload, dict):
            sc = payload.get("structuredContent")
            if sc is not None:
                try:
                    return json.dumps(sc, ensure_ascii=False, default=str)
                except Exception:
                    return str(sc)
            blocks = payload.get("content")
            if isinstance(blocks, list):
                texts = [
                    b.get("text") for b in blocks
                    if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
                ]
                if len(texts) == 1:
                    return texts[0]
                if texts:
                    return "\n".join(texts)
            try:
                return json.dumps(payload, ensure_ascii=False, default=str)
            except Exception:
                return str(payload)
        return str(payload)

    def _extract_resources_from_tool_result(
        self,
        result: ToolCallResult,
        *,
        task_id: str = "",
        session_id: str = "",
    ) -> List[Dict[str, Any]]:
        """从工具返回里提取可展示资源。

        优先级：
            1. 显式声明（result.result["resources"]）→ 走契约
            2. 隐式提取（MCP content / new_path / columns+rows）→ 走 fallback

        所有资源统一走 resource_store，返回统一模型 dict。
        """
        if not result or not getattr(result, "success", False):
            return []

        from core.resource.store import (
            save_bytes, save_base64, save_file, save_table, save_text,
        )

        resources: List[Dict[str, Any]] = []
        payload = result.result
        if not isinstance(payload, dict):
            return []

        tool_name = result.tool_name

        # ── 1. 显式声明（新契约） ──
        declared = payload.get("resources")
        if isinstance(declared, list) and declared:
            for item in declared:
                if not isinstance(item, dict):
                    continue
                    
                # ── 新增：已带 id + url 的资源（如 search_knowledge 返回的）直接透传 ──
                if item.get("id") and item.get("url"):
                    resources.append({
                        "id": item["id"],
                        "kind": item.get("kind") or "file",
                        "filename": item.get("filename") or f"{tool_name}_{item['id'][:8]}",
                        "mime": item.get("mime") or item.get("mime_type") or "",
                        "url": item["url"],
                        "source": item.get("source") or "rag",
                        "source_tool": tool_name,
                        "page_no": item.get("page_no"),
                        "size": item.get("size", 0),
                        # chunk_preview / chunk_filename 不下发前端
                    })
                    continue

                kind = item.get("kind")
                filename = item.get("filename") or f"{tool_name}_{kind}"
                mime = item.get("mime") or item.get("mime_type") or ""

                # 表格
                if kind == "table":
                    cols = item.get("columns")
                    rows = item.get("rows")
                    if isinstance(cols, list) and isinstance(rows, list):
                        res = save_table(
                            cols, rows,
                            filename=filename,
                            source="tool",
                            source_tool=tool_name,
                            task_id=task_id,
                            session_id=session_id,
                        )
                        resources.append(res.to_dict())
                    continue

                # 文件路径
                path = item.get("path") or item.get("file_path")
                if isinstance(path, str) and path:
                    res = save_file(
                        path,
                        filename=filename,
                        mime=mime,
                        source="tool",
                        source_tool=tool_name,
                        task_id=task_id,
                        session_id=session_id,
                    )
                    if res is not None:
                        resources.append(res.to_dict())
                    continue

                # bytes / base64 / 文本
                content = item.get("content")
                if isinstance(content, bytes):
                    res = save_bytes(
                        content,
                        filename=filename,
                        mime=mime,
                        source="tool",
                        source_tool=tool_name,
                        task_id=task_id,
                        session_id=session_id,
                    )
                    if res is not None:
                        resources.append(res.to_dict())
                elif isinstance(content, str):
                    if content.startswith("data:") or _looks_base64(content):
                        res = save_base64(
                            content,
                            filename=filename,
                            mime=mime,
                            source="tool",
                            source_tool=tool_name,
                            task_id=task_id,
                            session_id=session_id,
                        )
                    else:
                        res = save_text(
                            content,
                            filename=filename,
                            source="tool",
                            source_tool=tool_name,
                            task_id=task_id,
                            session_id=session_id,
                        )
                    if res is not None:
                        resources.append(res.to_dict())

            if resources:
                return _dedup_resources(resources)

        # ── 2. 隐式提取（fallback） ──

        # 2a0. search_knowledge 风格：chunks[i].resources
        chunks = payload.get("chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                chunk_resources = chunk.get("resources")
                if not isinstance(chunk_resources, list):
                    continue
                for item in chunk_resources:
                    if not isinstance(item, dict):
                        continue
                    rid = item.get("id")
                    url = item.get("url")
                    if not rid or not url:
                        continue
                    resources.append({
                        "id": rid,
                        "kind": item.get("kind") or "image",
                        "filename": item.get("filename") or "",
                        "mime": item.get("mime_type") or item.get("mime") or "",
                        "url": url,
                        "markdown": item.get("markdown") or "",
                        "page_no": item.get("page_no"),
                        "source": "rag",
                        "source_tool": tool_name,
                    })

        # 2a. 旧 resources 字段（已带 url）
        listed = payload.get("resources")
        if isinstance(listed, list):
            for item in listed:
                if isinstance(item, dict) and item.get("url"):
                    resources.append(item)

        # 2b. MCP content 数组
        content_blocks = payload.get("content")
        if isinstance(content_blocks, list):
            for idx, block in enumerate(content_blocks):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")

                if btype == "image":
                    data = block.get("data") or ""
                    mime = block.get("mimeType") or "image/png"
                    if not data:
                        continue
                    ext = mimetypes.guess_extension(mime) or ".png"
                    res = save_base64(
                        data,
                        filename=f"{tool_name}_{idx}{ext}",
                        mime=mime,
                        source="tool",
                        source_tool=tool_name,
                        task_id=task_id,
                        session_id=session_id,
                    )
                    if res is not None:
                        resources.append(res.to_dict())

                elif btype == "resource":
                    res_block = block.get("resource") or {}
                    uri = res_block.get("uri") or f"{tool_name}_{idx}"
                    filename = uri.rsplit("/", 1)[-1] or f"resource_{idx}"
                    mime = res_block.get("mimeType") or ""

                    blob = res_block.get("blob")
                    if blob:
                        res = save_base64(
                            blob,
                            filename=filename,
                            mime=mime,
                            source="tool",
                            source_tool=tool_name,
                            task_id=task_id,
                            session_id=session_id,
                        )
                        if res is not None:
                            resources.append(res.to_dict())
                        continue

                    text = res_block.get("text")
                    if text and isinstance(text, str):
                        res = save_text(
                            text,
                            filename=filename if "." in filename else f"{filename}.txt",
                            source="tool",
                            source_tool=tool_name,
                            task_id=task_id,
                            session_id=session_id,
                        )
                        if res is not None:
                            resources.append(res.to_dict())

        # 2c. 本地文件路径
        for key in ("new_path", "output_path"):
            path = payload.get(key)
            if not isinstance(path, str) or not path:
                continue
            res = save_file(
                path,
                source="tool",
                source_tool=tool_name,
                task_id=task_id,
                session_id=session_id,
            )
            if res is not None:
                resources.append(res.to_dict())

        # 2d. 表格（内置工具返回 columns+rows）
        columns = payload.get("columns")
        rows = payload.get("rows")
        if isinstance(columns, list) and isinstance(rows, list) and columns:
            res = save_table(
                columns, rows,
                filename=payload.get("filename") or tool_name,
                source="tool",
                source_tool=tool_name,
                task_id=task_id,
                session_id=session_id,
            )
            resources.append(res.to_dict())

        return _dedup_resources(resources)

    # ────────── 流式回答 ──────────

    async def _stream_answer(
        self,
        task_id: str,
        session_id: str,
        prior_messages: List[Dict[str, str]],
        message: str,
        mode: str,
        attachments: List[Dict[str, Any]],
        turn_id: str,
        rag_context: Optional[str] = None,
        rag_resources: Optional[List[Dict[str, Any]]] = None,
        tool_context: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        """串联 answer_stream 的 7 个模块。

        具体逻辑在 core/task/answer/ 下：
            multimodal.py / prompt_build.py / provider_call.py
            splitter_bridge.py / fallback_search.py
            asset_resolve.py / orchestrator.py
        """
        from core.task.answer import answer_stream

        async for evt in answer_stream(
            task_id=task_id,
            session_id=session_id,
            prior_messages=prior_messages,
            message=message,
            mode=mode,
            attachments=attachments,
            turn_id=turn_id,
            rag_context=rag_context,
            rag_resources=rag_resources,
            tool_context=tool_context,
            make_step=self._step,
            emit_step=self._emit_step,
            session_provider=self._session_provider,
            rag_provider=self._rag_provider,
            llm_provider=self._llm_provider,
            multimodal_provider=self._multimodal_provider,
            agent_provider=self._agent_provider,
            persist_task_status=self._persist_task_status,
        ):
            yield evt


    async def _execute_tool(
        self,
        task_id: str,
        session_id: str,
        tool_call: ToolCallRequest,
        turn_id: str,
        step_id: Optional[str] = None,
        tool_execution_context: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        step_id = step_id or f"{task_id}:{turn_id}:tool:{uuid.uuid4().hex[:8]}"
        label = f"执行工具：{tool_call.tool_name}"
        logger.info(
            "[dsh][task_service] _execute_tool start: %s params=%r",
            tool_call.tool_name, tool_call.parameters,
        )

        running_step = self._step(
            step_id, label, "running", "action",
            path=self._tool_path(tool_call),
        )
        yield self._emit_step(task_id, running_step)

        try:
            result = await self._agent_provider().execute_tool_async(tool_call)
            logger.info(
                "[dsh][task_service] _execute_tool done: %s success=%r error=%r",
                tool_call.tool_name, result.success, result.error,
            )
        except Exception:
            logger.exception(
                "[dsh][task_service] _execute_tool raised: %s",
                tool_call.tool_name,
            )
            raise

        log = self._agent_provider().format_tool_result(result)
        status = "done" if result.success else "failed"
        logger.info(
            "[dsh][task_service] _execute_tool formatted: %s status=%s log=%r",
            tool_call.tool_name, status, log[:300],
        )

        done_step = self._step(
            step_id,
            label if result.success else f"{label}失败",
            status,
            "action",
            path=self._tool_path(tool_call),
            log=log,
        )
        yield self._emit_step(task_id, done_step)

        file_log = self._file_log_entry(tool_call, result)
        if file_log:
            yield self._emit_file_log(session_id, file_log)

        # ── 提取二进制资源 ──
        try:
            found = self._extract_resources_from_tool_result(
                result,
                task_id=task_id,
                session_id=session_id,
            )
            if found:
                yield self._emit_step(
                    task_id,
                    self._step(
                        f"{task_id}:{turn_id}:resources:{uuid.uuid4().hex[:6]}",
                        f"工具产生 {len(found)} 个资源",
                        "done",
                        "resource",
                        resources=found,
                    ),
                )
        except Exception:
            logger.exception("提取工具资源失败")

        # 把结果累积到 tool_execution_context（如果调用方传了）
        # 这样外层不再需要重复 execute_tool_async
        if tool_execution_context is not None:
            try:
                if result.success and result.result is not None:
                    handle = build_handle_summary(
                        task_id=task_id,
                        step_index=0,
                        tool_name=tool_call.tool_name,
                        payload=result.result,
                        max_inline_tokens=4000,
                    )
                    if handle.get("inline"):
                        payload_text = self._compact_tool_payload(result.result)
                        tool_execution_context.append(
                            f"【工具 {tool_call.tool_name} 返回】\n{payload_text[:8000]}"
                        )
                    else:
                        summary_text = json.dumps(
                            handle.get("summary") or {}, ensure_ascii=False, default=str
                        )
                        tool_execution_context.append(
                            f"【工具 {tool_call.tool_name} 返回】\n"
                            f"⚠️ 完整结果已卸载，result_id=`{handle.get('result_id')}`\n"
                            f"调用 `read_tool_result` 按需读取。\n\n{summary_text[:4000]}"
                        )
            except Exception:
                logger.exception("累积工具结果失败")

        self._session_provider().add_message(session_id, "assistant", log)
        final_status = "done" if result.success else "failed"
        self._persist_task_status(task_id, final_status)
        yield sse_event("done", task_status=final_status)

    # ────────── orchestrator 事件 → SSE ──────────

    async def _process_orchestrator_event(
        self,
        event: Dict[str, Any],
        *,
        task_id: str,
        session_id: str,
        turn_id: str,
        message: str,
        mode: str,
        workspace: Optional[List[str]],
        workspace_trusted: bool,
        auto_discover_mcp: bool,
        analysis_step_id: str,
        tool_execution_context: List[str],
        accumulated_resources: List[Dict[str, Any]],
        event_prefix: str = "orch",
    ) -> AsyncIterator[str]:
        """处理单个 orchestrator 事件，产出 SSE。

        调用方负责维护 tool_execution_context / accumulated_resources，
        并在 approval_required 后退出循环（见调用方）。
        """
        event_type = event.get("type")

        if event_type == "thinking":
            thought = (event.get("thought") or "").strip()
            if thought:
                log_text = (event.get("log") or thought).strip()
                yield self._emit_step(
                    task_id,
                    self._step(
                        f"{task_id}:{turn_id}:{event_prefix}-think:{event['step']}",
                        thought[:120],
                        "done",
                        "think",
                        log=log_text,
                    ),
                )

        elif event_type == "approval_required":
            tool_call = event["tool_call"]
            reason = event.get("reason") or "需要用户确认"
            # ── 优先用 orchestrator 网关生成的 approval_id（params_hash 绑定） ──
            approval_id = event.get("approval_id") or f"approval-{uuid.uuid4().hex}"
            tool_step_id = f"{task_id}:{turn_id}:approval:{approval_id}"
            title = f"确认执行工具：{tool_call.tool_name}（{reason}）"

            pending = PendingApproval(
                task_id=task_id,
                session_id=session_id,
                step_id=tool_step_id,
                approval_id=approval_id,
                tool_call=tool_call,
                title=title,
                turn_id=turn_id,
                path=self._tool_path(tool_call),
                goal=message,
                mode=mode,
                workspace=workspace,
                workspace_trusted=workspace_trusted,
                auto_discover_mcp=auto_discover_mcp,
                step_index=event.get("step", 1),
                state_snapshot=event.get("state_snapshot"),
            )
            self._store_pending(pending)

            approval = {
                "id": approval_id,
                "title": title,
                "toolName": tool_call.tool_name,
                "parameters": tool_call.parameters,
                "dangerLevel": "medium",
                "path": pending.path,
                "taskId": task_id,
                "stepId": tool_step_id,
                # 前端 approve 时必须回传 session_id（后端 Form 必填）
                "sessionId": session_id,
            }
            waiting_step = self._step(
                tool_step_id, title, "waiting", "action",
                path=pending.path, approval=approval,
            )
            yield self._emit_step(
                task_id,
                self._step(analysis_step_id, "任务分析完成", "done", "think"),
            )
            yield self._emit_step(task_id, waiting_step)
            self._persist_task_status(task_id, "waiting")
            yield sse_event("done", task_status="waiting")

        elif event_type == "tool_start":
            tool_call = event["tool_call"]
            yield self._emit_step(
                task_id,
                self._step(
                    f"{task_id}:{turn_id}:{event_prefix}-tool:{event['step']}",
                    f"调用工具：{tool_call.tool_name}",
                    "running",
                    "action",
                    path=self._tool_path(tool_call),
                ),
            )

        elif event_type == "tool_result":
            tool_call = event["tool_call"]
            result = event["result"]
            status = "done" if result.success else "failed"
            label = (
                f"调用工具：{tool_call.tool_name}"
                if result.success
                else f"调用工具：{tool_call.tool_name}（失败）"
            )

            # ── 1. 提取资源（必须用原始 bytes） ──
            try:
                found = self._extract_resources_from_tool_result(
                    result,
                    task_id=task_id,
                    session_id=session_id,
                )
                if found:
                    # 跨多次工具调用去重（同一 rid 只保留一次）
                    existing_ids = {r.get("id") for r in accumulated_resources if r.get("id")}
                    new_items = [r for r in found if r.get("id") and r["id"] not in existing_ids]
                    if new_items:
                        accumulated_resources.extend(new_items)
                        logger.info(
                            "[dsh][task_service] appended %d new resources (skipped %d dups)",
                            len(new_items), len(found) - len(new_items),
                        )
            except Exception:
                logger.exception("提取工具资源失败")

            # ── 2. 剥离 bytes（防止污染 json） ──
            if isinstance(getattr(result, "result", None), dict):
                result.result = self._strip_bytes(result.result)

            # event["log"] 来自 agent.format_tool_result(result)，
            # 已包含 structuredContent / content / leftover / 错误原文，
            # 不再追加 result.result（此前会导致"原始返回 + 一模一样副本"）。
            log_parts: List[str] = []
            if event.get("log"):
                log_parts.append(str(event["log"]))

            # 仅当 format 没覆盖到错误时补一句（避免重复）
            if not result.success and getattr(result, "error", None):
                err_text = f"【错误】{result.error}"
                head = log_parts[0] if log_parts else ""
                if err_text not in head and str(result.error) not in head:
                    log_parts.append(err_text)

            combined_log = "\n\n".join(log_parts) if log_parts else None

            yield self._emit_step(
                task_id,
                self._step(
                    f"{task_id}:{turn_id}:{event_prefix}-tool:{event['step']}",
                    label,
                    status,
                    "action",
                    path=self._tool_path(tool_call),
                    log=combined_log,
                ),
            )

            if result.result is not None:
                try:
                    tag = "成功" if result.success else "失败"
                    err_note = (
                        f"\n错误: {result.error}"
                        if not result.success and result.error
                        else ""
                    )

                    handle = build_handle_summary(
                        task_id=task_id,
                        step_index=event["step"],
                        tool_name=tool_call.tool_name,
                        payload=result.result,
                        max_inline_tokens=2000,
                    )

                    if handle.get("inline"):
                        # 小结果：只保留"产出型"字段，中间过程 log 已在 state.history
                        _PRODUCTIVE_KEYS = (
                            "new_path", "output_path", "file_path",
                            "resources", "columns", "rows", "url",
                            "content", "content_preview", "text",
                            "message", "summary",
                        )
                        if isinstance(result.result, dict):
                            productive = {
                                k: v for k, v in result.result.items()
                                if k in _PRODUCTIVE_KEYS
                            }
                            if productive:
                                payload_text = json.dumps(
                                    productive, ensure_ascii=False, default=str,
                                )
                            else:
                                payload_text = self._compact_tool_payload(result.result)
                        else:
                            payload_text = self._compact_tool_payload(result.result)
                        tool_execution_context.append(
                            f"【工具 {tool_call.tool_name} 返回（{tag}）】"
                            f"{err_note}\n{payload_text[:3000]}"
                        )
                    else:
                        # 大结果：句柄 + 摘要
                        summary = handle.get("summary")
                        if summary:
                            summary_text = json.dumps(
                                summary, ensure_ascii=False, default=str
                            )
                        else:
                            summary_text = "（摘要不可用，请用 read_tool_result 读取）"

                        tool_execution_context.append(
                            f"【工具 {tool_call.tool_name} 返回（{tag}）】{err_note}\n"
                            f"⚠️ 完整结果过大（约 {handle.get('estimated_tokens')} tokens），"
                            f"已卸载到磁盘。\n"
                            f"result_id: `{handle.get('result_id')}`\n"
                            f"handle: `{handle.get('handle_path')}`\n"
                            f"如需完整内容，调用 `read_tool_result` 工具，"
                            f"参数 result_id=`{handle.get('result_id')}`、task_id=`{task_id}`，"
                            f"可指定 filename 和 start_index 分页读取。\n\n"
                            f"## 摘要（每个文档采样 {len((summary or {}).get('files', []))} 个文档）\n"
                            f"{summary_text[:2000]}"
                        )
                except Exception:
                    logger.exception("累积工具结果失败")

            # ── web 结果按来源类型分层，注入 context（供最终回答引用） ──
            _is_web_tool = (
                tool_call.tool_name.startswith("mcp__tavily")
                or tool_call.tool_name == "fetch_url"
                or tool_call.tool_name.startswith("mcp__fetch")
            )
            if _is_web_tool and isinstance(result.result, dict):
                try:
                    from tools.orchestration.core import AgentOrchestrator
                    _buckets = AgentOrchestrator._bucket_web_results(result.result)
                    if _buckets:
                        _labels = AgentOrchestrator._TIER_LABELS
                        _lines = [
                            "## 搜索结果按来源类型分层（供引用/判断权威度）"
                        ]
                        for b in ("official", "news", "forum", "blog", "unknown"):
                            if b not in _buckets:
                                continue
                            _lines.append(
                                f"\n### {_labels.get(b, b)}（{len(_buckets[b])} 条）"
                            )
                            for it in _buckets[b][:5]:
                                _title = it.get("title") or it.get("url") or ""
                                _lines.append(
                                    f"- [{_title}]({it['url']})\n"
                                    f"  {str(it.get('content') or '')[:300]}"
                                )
                        _lines.append(
                            "\n**引用规则**：official/news 可作为事实依据并注明来源；"
                            "forum/blog 仅作为补充细节或争议观点，"
                            "必须标注“来自社区讨论/个人博客”，不得当作权威事实。"
                            "不同层级来源说法矛盾时，并列呈现并标注来源类型。"
                        )
                        tool_execution_context.append("\n".join(_lines))
                except Exception:
                    logger.exception("web source bucketing failed")
                    
            file_log = self._file_log_entry(tool_call, result)
            if file_log:
                yield self._emit_file_log(session_id, file_log)
                # 删除动作且成功 → 额外发 trash 事件
                if file_log["action"] == "delete" and file_log["status"] == "success":
                    trash_id = None
                    trash_path = None
                    if isinstance(result.result, dict):
                        trash_id = result.result.get("trash_id")
                        trash_path = result.result.get("new_path")
                    yield sse_event("trash", item={
                        "id": trash_id or file_log["id"],
                        "original_path": file_log["path"],
                        "original_name": os.path.basename(file_log["path"]),
                        "trash_path": trash_path or "",
                        "deleted_at": datetime.now().isoformat(),
                        "size": 0,
                    })

            # ── 落盘工具执行结果（供事后回放） ──
            try:
                from infra.debug_log.llm import dump_llm_call
                _debug = None
                if isinstance(result.result, dict):
                    _debug = result.result.get("_debug")
                dump_llm_call(
                    kind="tool_exec",
                    prompt="",
                    response=json.dumps(
                        self._strip_bytes(result.result),
                        ensure_ascii=False, default=str,
                    ) if result.result is not None else None,
                    tool_name=tool_call.tool_name,
                    duration_ms=(result.execution_time * 1000
                                 if result.execution_time else None),
                    error=result.error,
                    extra={
                        "step": event.get("step"),
                        "params": tool_call.parameters,
                        "success": result.success,
                        "debug": _debug,
                        "error_type": type(result.error).__name__
                            if result.error else None,
                    },
                )
            except Exception:
                logger.exception("[dsh][task_service] dump tool_exec failed")

        elif event_type == "done":
            reason = event.get("reason", "done")
            detail = event.get("detail", "")
            label = {
                "done": "任务分析完成",
                "partial": f"部分完成：{detail}" if detail else "部分完成",
                "blocked": f"外部阻塞：{detail}" if detail else "外部阻塞",
                "quality_fail": f"校验未通过：{detail}" if detail else "校验未通过",
            }.get(reason, "任务分析完成")
            status = "done" if reason == "done" else "failed"
            yield self._emit_step(
                task_id,
                self._step(analysis_step_id, label, status, "think"),
            )
            try:
                memory = self._session_provider()
                if hasattr(memory, "update_task"):
                    memory.update_task(
                        task_id, stop_reason=reason, stop_detail=detail,
                    )
            except Exception:
                logger.exception("持久化 stop_reason 失败: %s", task_id)

        elif event_type == "max_steps":
            yield self._emit_step(
                task_id,
                self._step(
                    analysis_step_id,
                    f"已达最大步数 {event['steps']}",
                    "done",
                    "think",
                ),
            )
            # 持久化 stop_reason，供 _stream_answer 拼【终止原因】
            try:
                memory = self._session_provider()
                if hasattr(memory, "update_task"):
                    memory.update_task(
                        task_id,
                        stop_reason="max_steps",
                        stop_detail=f"已达最大步数 {event['steps']}",
                    )
            except Exception:
                logger.exception("持久化 max_steps 失败: %s", task_id)

        elif event_type == "error":
            yield sse_event(
                "error", message=event.get("message", "编排失败")
            )
            yield self._emit_step(
                task_id,
                self._step(
                    analysis_step_id, "任务分析失败", "failed", "think"
                ),
            )


    # ────────── 主入口 ──────────

    async def stream_task(
        self,
        *,
        task_id: Optional[str],
        message: str,
        history: List[Dict[str, str]],
        mode: str,
        workspace: Optional[List[str]],
        session_id: Optional[str],
        auto_discover_mcp: bool = False,
        attachments: List[Dict[str, Any]],
    ) -> AsyncIterator[str]:
        task_id = task_id or f"task-{uuid.uuid4().hex}"
        turn_id = uuid.uuid4().hex[:8]

        try:
            mode = validate_mode(mode)
        except ValueError as exc:
            yield sse_event("error", message=str(exc))
            return

        session_id = self._ensure_session(session_id, message)

        try:
            self._agent_provider().clear_available_tools()
        except Exception:
            logger.exception("clear_available_tools failed")

        begin_turn(session_id=session_id, task_id=task_id, question=message)

        try:
            from infra.debug_log.mem0 import begin_task as begin_mem0_task
            begin_mem0_task(session_id=session_id, task_id=task_id)
        except Exception:
            logger.exception("begin_mem0_task failed")

        with set_turn_context(session_id=session_id, task_id=task_id, question=message):
          _sec_ctx_token = None
          try:
            memory = self._session_provider()
            prior_messages = (
                memory.get_conversation_context(session_id, 20)
                if session_id else history
            )
            if not prior_messages:
                prior_messages = history
            self._record_session_input(session_id, message, attachments)

            try:
                from core.memory.mem0_layer import remember_user_message
                remember_user_message(session_id, message)
            except Exception:
                logger.exception("remember_user_message 失败")

            self._persist_task(
                task_id=task_id, session_id=session_id, message=message,
                mode=mode, workspace=workspace, attachments=attachments,
            )

            yield sse_event("task", task_id=task_id, session_id=session_id, turn_id=turn_id)

            if message.strip():
                yield self._emit_step(
                    task_id,
                    self._step(f"{task_id}:{turn_id}:user", message.strip(), "done", "user"),
                )

            for index, attachment in enumerate(attachments):
                yield self._emit_step(
                    task_id,
                    self._step(
                        f"{task_id}:{turn_id}:attachment:{index}",
                        f"接收附件：{attachment_label(attachment)}",
                        "done", "action", path=attachment_label(attachment),
                    ),
                )

            analysis_step_id = f"{task_id}:{turn_id}:analysis"
            yield self._emit_step(
                task_id,
                self._step(analysis_step_id, "正在理解任务…", "running", "think"),
            )

            if mode == MODE_ASK:
                allowed_dangers: set = {"safe"}
                workspace_roots_for_ctx: Optional[List[str]] = None
                workspace_trusted = False
            else:
                allowed_dangers = {"safe", "medium", "high"}
                workspace_roots_for_ctx = self._normalise_workspace(workspace) or None
                workspace_trusted = bool(workspace_roots_for_ctx)

            from tools.security.gateway import (
                SecurityContext, set_security_context, reset_security_context,
            )
            _sec_ctx_token = set_security_context(SecurityContext(
                allowed_danger_levels=allowed_dangers,
                workspace_roots=workspace_roots_for_ctx,
                workspace_trusted=workspace_trusted,
                session_id=session_id,
            ))

            accumulated_resources: List[Dict[str, Any]] = []
            accumulated_resources.extend(self._collect_session_resources(session_id))
            tool_execution_context: List[str] = []

            if mode == MODE_ASK:
                # ── ask 模式：只读，但必须能获取信息 ──
                # 设计：
                #   1. 仍然只允许 safe 级工具（不能写/删/执行）
                #   2. 不允许 MCP discovery（ask 不主动扩展能力）
                #   3. 走 orchestrator，让模型自主决定是否调用只读搜索工具
                #   4. 如果 orchestrator 不可用，退化为“注入只读工具目录”直答
                yield self._emit_step(
                    task_id,
                    self._step(analysis_step_id, "对话模式（只读检索）", "done", "think"),
                )

                ask_allowed_dangers = {"safe"}
                ask_workspace_roots = None
                ask_workspace_trusted = False

                used_ask_orchestrator = False
                if self._orchestrator_provider is not None:
                    used_ask_orchestrator = True
                    orchestrator = self._orchestrator_provider()
                    ask_tool_context: List[str] = []
                    ask_resources: List[Dict[str, Any]] = []
                    async for event in orchestrator.run(
                        message,
                        allowed_danger_levels=ask_allowed_dangers,
                        allow_mcp_discovery=False,
                        workspace_paths=ask_workspace_roots,
                        prior_messages=prior_messages,
                        session_id=session_id,
                        task_id=task_id,
                    ):
                        async for evt in self._process_orchestrator_event(
                            event,
                            task_id=task_id, session_id=session_id, turn_id=turn_id,
                            message=message, mode=mode, workspace=None,
                            workspace_trusted=ask_workspace_trusted,
                            auto_discover_mcp=False,
                            analysis_step_id=analysis_step_id,
                            tool_execution_context=ask_tool_context,
                            accumulated_resources=ask_resources,
                            event_prefix="ask",
                        ):
                            yield evt
                        if event.get("type") == "approval_required":
                            # ask 模式不应产生写审批；防御性返回
                            return

                # 无论 orchestrator 是否可用，最终都基于已获取的工具结果生成回答
                async for evt in self._stream_answer(
                    task_id, session_id, prior_messages, message, mode,
                    attachments, turn_id,
                    rag_context=None,
                    rag_resources=None,
                    tool_context=ask_tool_context if used_ask_orchestrator else None,
                ):
                    yield evt
                return

            used_orchestrator = False
            if self._orchestrator_provider is not None:
                used_orchestrator = True
                orchestrator = self._orchestrator_provider()
                async for event in orchestrator.run(
                    message,
                    allowed_danger_levels=allowed_dangers,
                    allow_mcp_discovery=auto_discover_mcp,
                    workspace_paths=workspace_roots_for_ctx,
                    prior_messages=prior_messages,
                    session_id=session_id,
                    task_id=task_id,
                ):
                    async for evt in self._process_orchestrator_event(
                        event,
                        task_id=task_id, session_id=session_id, turn_id=turn_id,
                        message=message, mode=mode, workspace=workspace,
                        workspace_trusted=workspace_trusted,
                        auto_discover_mcp=auto_discover_mcp,
                        analysis_step_id=analysis_step_id,
                        tool_execution_context=tool_execution_context,
                        accumulated_resources=accumulated_resources,
                        event_prefix="orch",
                    ):
                        yield evt
                    if event.get("type") == "approval_required":
                        return

            if not used_orchestrator:
                tool_call: Optional[ToolCallRequest] = None
                try:
                    attachment_names = ", ".join(
                        attachment_label(item) for item in attachments
                    )
                    intent_message = message
                    if attachment_names:
                        intent_message += f"\n当前附件：{attachment_names}"
                    intent = self._agent_provider().analyze_intent(intent_message, use_llm=True)
                    tool_call = self._agent_provider().prepare_tool_call(intent)
                except Exception as error:
                    yield sse_event("error", message=f"任务分析失败：{error}")

                if tool_call:
                    needs_confirmation = (
                        tool_call.user_confirmation_needed or not workspace_trusted
                    )
                    yield self._emit_step(
                        task_id,
                        self._step(
                            analysis_step_id,
                            f"已识别操作：{tool_call.tool_name}",
                            "done", "think",
                        ),
                    )
                    if needs_confirmation:
                        approval_id = f"approval-{uuid.uuid4().hex}"
                        tool_step_id = f"{task_id}:{turn_id}:approval:{approval_id}"
                        title = f"确认执行工具：{tool_call.tool_name}"
                        pending = PendingApproval(
                            task_id=task_id, session_id=session_id,
                            step_id=tool_step_id, approval_id=approval_id,
                            tool_call=tool_call, title=title, turn_id=turn_id,
                            path=self._tool_path(tool_call), mode=mode,
                            workspace=workspace, workspace_trusted=workspace_trusted,
                            auto_discover_mcp=auto_discover_mcp,
                        )
                        self._store_pending(pending)

                        approval = {
                            "id": approval_id, "title": title,
                            "toolName": tool_call.tool_name,
                            "parameters": tool_call.parameters,
                            "dangerLevel": "high" if tool_call.user_confirmation_needed else "medium",
                            "path": pending.path, "taskId": task_id, "stepId": tool_step_id,
                        }
                        waiting_step = self._step(
                            tool_step_id, title, "waiting", "action",
                            path=pending.path, approval=approval,
                        )
                        yield self._emit_step(task_id, waiting_step)
                        self._persist_task_status(task_id, "waiting")
                        yield sse_event("done", task_status="waiting")
                        return

                    async for event in self._execute_tool(
                        task_id, session_id, tool_call, turn_id,
                        tool_execution_context=tool_execution_context,
                    ):
                        yield event
                    return

                yield self._emit_step(
                    task_id,
                    self._step(analysis_step_id, "已完成任务分析", "done", "think"),
                )

            if accumulated_resources:
                yield self._emit_step(
                    task_id,
                    self._step(
                        f"{task_id}:{turn_id}:resources",
                        f"相关资源 {len(accumulated_resources)} 个",
                        "done",
                        "resource",
                        resources=accumulated_resources,
                    ),
                )

            rag_context: Optional[str] = None
            rag_resources: Optional[List[Dict[str, Any]]] = [
                r for r in accumulated_resources
                if r.get("source_tool") == "search_knowledge"
                or r.get("source") == "rag"
            ]

            async for event in self._stream_answer(
                task_id,
                session_id,
                prior_messages,
                message,
                mode,
                attachments,
                turn_id,
                rag_context=rag_context,
                rag_resources=rag_resources,
                tool_context=tool_execution_context,
            ):
                yield event
          finally:
            # ── 复位 SecurityContext（必须最先） ──
            if _sec_ctx_token is not None:
                try:
                    reset_security_context(_sec_ctx_token)
                except Exception:
                    logger.exception("reset_security_context failed")

            end_turn()
            try:
                from infra.debug_log.mem0 import end_task as end_mem0_task
                end_mem0_task()
            except Exception:
                logger.exception("end_mem0_task failed")

            # 若任务尚未进入终态（done/failed/waiting），标记为 cancelled。
            # 覆盖"客户端断开导致生成器被取消"的场景，避免僵尸 running。
            try:
                memory = self._session_provider()
                task = memory.get_task(task_id) if hasattr(memory, "get_task") else None
                if task and task.get("status") in {"running", "pending", None}:
                    self._persist_task_status(task_id, "cancelled")
                    logger.info(
                        "[dsh][task_service] task %s marked cancelled (stream ended)",
                        task_id,
                    )
            except Exception:
                logger.exception("标记任务 cancelled 失败: %s", task_id)

    # ────────── 审批处理 ──────────

    async def stream_approval(
        self,
        task_id: str,
        approval_id: str,
        action: str,
    ) -> AsyncIterator[str]:
        with self._lock:
            logger.info(
                "[dsh][task_service] stream_approval: task_id=%r approval_id=%r "
                "pending_keys=%r",
                task_id, approval_id, list(self._pending.keys()),
            )
        pending = self._take_pending(approval_id, task_id)
        if pending is None:
            with self._lock:
                _exists = approval_id in self._pending
                _owner = (
                    self._pending[approval_id].task_id
                    if _exists else None
                )
            logger.warning(
                "[dsh][task_service] stream_approval stale: approval_id=%r "
                "task_id=%r exists=%r owner_task=%r",
                approval_id, task_id, _exists, _owner,
            )
            try:
                from infra.debug_log.llm import dump_llm_call
                dump_llm_call(
                    kind="approval_stale",
                    prompt="",
                    response=None,
                    extra={
                        "approval_id": approval_id,
                        "task_id": task_id,
                        "exists": _exists,
                        "owner_task": _owner,
                    },
                )
            except Exception:
                logger.exception("[dsh][task_service] dump approval_stale failed")
            yield sse_event("done", task_status="stale")
            return

        # ── 设置当前 turn 上下文（供 dump_llm_call 用）──
        from infra.debug_log.llm import begin_turn, end_turn
        begin_turn(
            session_id=pending.session_id,
            task_id=task_id,
            question=f"approval-{approval_id}",
        )
        try:
            # ── 拿到 pending：落盘 approval_take ──
            try:
                from infra.debug_log.llm import dump_llm_call
                dump_llm_call(
                    kind="approval_take",
                    prompt="",
                    response=None,
                    tool_name=pending.tool_call.tool_name,
                    extra={
                        "approval_id": approval_id,
                        "task_id": task_id,
                        "session_id": pending.session_id,
                        "tool_name": pending.tool_call.tool_name,
                        "action": action,
                        "params": pending.tool_call.parameters,
                    },
                )
            except Exception:
                logger.exception("[dsh][task_service] dump approval_take failed")

            # ── 同步 ApprovalStore 决策（供网关 verify 用） ──
            try:
                from tools.security.approval import get_approval_store
                get_approval_store().decide(approval_id, pending.session_id, action)
            except Exception:
                logger.exception("同步 ApprovalStore.decide 失败: %s", approval_id)

            # ── 落盘审批决策 ──
            try:
                from infra.debug_log.llm import dump_llm_call
                dump_llm_call(
                    kind="approval_decision",
                    prompt="",
                    response=None,
                    tool_name=pending.tool_call.tool_name,
                    extra={
                        "approval_id": approval_id,
                        "task_id": task_id,
                        "session_id": pending.session_id,
                        "action": action,
                        "allowed": action in {"allow", "allow-always"},
                    },
                )
            except Exception:
                logger.exception("[dsh][task_service] dump approval_decision failed")

            if action not in {"allow", "allow-always"}:
                rejected = self._step(
                    pending.step_id,
                    f"已拒绝：{pending.title}",
                    "failed",
                    "action",
                    path=pending.path,
                )
                yield self._emit_step(task_id, rejected)

                self._session_provider().add_message(
                    pending.session_id,
                    "assistant",
                    f"已拒绝工具操作：{pending.tool_call.tool_name}",
                )
                self._persist_task_status(task_id, "failed")
                yield sse_event("done", task_status="failed")
                return

            # ── 执行工具（内联，避免走 _execute_tool 的抽象层） ──
            tool_call = pending.tool_call
            step_id = pending.step_id
            label = f"执行工具：{tool_call.tool_name}"

            # 审批路径独立，需自己维护 accumulated_resources
            accumulated_resources: List[Dict[str, Any]] = []

            yield self._emit_step(
                task_id,
                self._step(
                    step_id,
                    label,
                    "running",
                    "action",
                    path=self._tool_path(tool_call),
                ),
            )

            logger.info(
                "[dsh][task_service] approval executing: %s params=%r",
                tool_call.tool_name, tool_call.parameters,
            )


            # ── 新增：注入 approval_id，供 execute_tool_async 的 verify 用 ──
            try:
                from tools.security.gateway import get_security_context
                _ctx = get_security_context()
                if _ctx is not None:
                    _ctx.approved_approval_ids.add(approval_id)
            except Exception:
                logger.exception("注入 approved_approval_ids 失败: %s", approval_id)

            tool_call.parameters = dict(tool_call.parameters or {})
            tool_call.parameters["__approval_id__"] = approval_id

            try:
                result = await self._agent_provider().execute_tool_async(tool_call)
            except Exception as error:
                logger.exception(
                    "[dsh][task_service] approval execution raised: %s",
                    tool_call.tool_name,
                )
                from tools.types import ToolCallResult
                result = ToolCallResult(
                    tool_name=tool_call.tool_name,
                    success=False,
                    error=str(error),
                )

            log = self._agent_provider().format_tool_result(result)
            status = "done" if result.success else "failed"
            logger.info(
                "[dsh][task_service] approval executed: %s\n"
                "  params=%r\n"
                "  success=%r\n"
                "  error=%r\n"
                "  result_type=%s\n"
                "  result_preview=%r",
                tool_call.tool_name,
                tool_call.parameters,
                result.success,
                result.error,
                type(result.result).__name__,
                (
                    str(result.result)[:500]
                    if result.result is not None
                    else None
                ),
            )

            # ── 先提取资源（需要读原始 bytes） ──
            try:
                found = self._extract_resources_from_tool_result(
                    result,
                    task_id=task_id,
                    session_id=pending.session_id,
                )
                if found:
                    # 跨多次工具调用去重（同一 rid 只保留一次）
                    existing_ids = {r.get("id") for r in accumulated_resources if r.get("id")}
                    new_items = [r for r in found if r.get("id") and r["id"] not in existing_ids]
                    if new_items:
                        accumulated_resources.extend(new_items)
                    yield self._emit_step(
                        task_id,
                        self._step(
                            f"{task_id}:{step_id}:resources",
                            f"工具产生 {len(new_items)} 个新资源",
                            "done",
                            "resource",
                            resources=new_items,
                        ),
                    )
            except Exception:
                logger.exception("提取工具资源失败")

            # ── 再剥离 bytes（防止污染 context / json） ──
            if isinstance(getattr(result, "result", None), dict):
                result.result = self._strip_bytes(result.result)

            # 审批路径没有 orchestrator event；log 已是 format_tool_result 的结果，
            # 已包含 structuredContent / content / 错误原文，不再追加 result.result。
            log_parts: List[str] = []
            if log:
                log_parts.append(str(log))

            if not result.success and getattr(result, "error", None):
                err_text = f"【错误】{result.error}"
                head = log_parts[0] if log_parts else ""
                if err_text not in head and str(result.error) not in head:
                    log_parts.append(err_text)

            combined_log = "\n\n".join(log_parts) if log_parts else None

            yield self._emit_step(
                task_id,
                self._step(
                    step_id,
                    label if result.success else f"{label}（失败）",
                    status,
                    "action",
                    path=self._tool_path(tool_call),
                    log=combined_log,
                ),
            )

            file_log = self._file_log_entry(tool_call, result)
            if file_log:
                yield self._emit_file_log(pending.session_id, file_log)
                # 删除动作且成功 → 额外发 trash 事件
                if file_log["action"] == "delete" and file_log["status"] == "success":
                    trash_id = None
                    trash_path = None
                    if isinstance(result.result, dict):
                        trash_id = result.result.get("trash_id")
                        trash_path = result.result.get("new_path")
                    yield sse_event("trash", item={
                        "id": trash_id or file_log["id"],
                        "original_path": file_log["path"],
                        "original_name": os.path.basename(file_log["path"]),
                        "trash_path": trash_path or "",
                        "deleted_at": datetime.now().isoformat(),
                        "size": 0,
                    })

            # ── 无法恢复 orchestrator → 直接结束 ──
            if (
                pending.state_snapshot is None
                or self._orchestrator_provider is None
            ):
                final_status = "done" if result.success else "failed"
                self._session_provider().add_message(
                    pending.session_id, "assistant", log
                )
                self._persist_task_status(task_id, final_status)
                yield sse_event("done", task_status=final_status)
                return

            # ── 注入结果到 state，恢复 orchestrator ──
            state = OrchestratorState.from_snapshot(pending.state_snapshot)
            step_num = pending.step_index

            # 通用：审批路径也要从错误里回填 known_enums
            # （run.py 的 tool_result 分支只覆盖非审批路径，审批后执行的结果流经这里）
            try:
                if state is not None:
                    orch = self._orchestrator_provider() if self._orchestrator_provider else None
                    extractor = getattr(orch, "_extract_enum_from_error", None) if orch else None
                    if extractor is None:
                        # 兜底：直接用 RunMixin 的静态方法
                        from tools.orchestration.run import RunMixin
                        extractor = RunMixin._extract_enum_from_error
                    _enum_map = extractor(result)
                    if _enum_map:
                        for _tn, _vals in _enum_map.items():
                            state.known_enums[_tn] = _vals
                            logger.info(
                                "[dsh][task_service] learned %d enum values for %s from error (approval path): %r",
                                len(_vals), _tn, _vals[:10],
                            )
                    # 通用：检测工具级不可用
                    detect = getattr(orch, "_detect_unavailable_from_error", None) if orch else None
                    if detect is None:
                        from tools.orchestration.run import RunMixin
                        detect = RunMixin._detect_unavailable_from_error
                    _unavail_reason = detect(result)
                    if _unavail_reason:
                        _tn = getattr(result, "tool_name", None)
                        if _tn:
                            state.unavailable_tools[_tn] = _unavail_reason
                            logger.warning(
                                "[dsh][task_service] tool %s auto-marked unavailable (approval path): %s",
                                _tn, _unavail_reason[:100],
                            )
            except Exception:
                logger.exception("[dsh][task_service] extract enum / unavailable from error failed (approval path)")

            if result.success:
                state.mark_success(tool_call.tool_name, tool_call.parameters, step_num)
                state.add_history(
                    f"Step {step_num}: {tool_call.tool_name}({tool_call.parameters}) "
                    f"→ OK: {log[:400]}"
                )

                # ── Mem0：审批后成功也写事实 ──
                try:
                    from core.memory.mem0_layer import remember_tool_success
                    _meta_for_origin = self._agent_provider().list_all_tools().get(
                        tool_call.tool_name
                    )
                    _origin = getattr(_meta_for_origin, "origin", None) or "builtin"
                    remember_tool_success(
                        session_id=state.session_id or pending.session_id,
                        tool_name=tool_call.tool_name,
                        params=tool_call.parameters,
                        result=result.result,
                        origin=_origin,
                    )
                except Exception:
                    logger.exception("remember_tool_success (approval) 失败")
            else:
                state.mark_failure(
                    tool_call.tool_name, tool_call.parameters, result.error or ""
                )
                # ── 补：更新连续失败计数，供 _intercept_call 拦截重复调用 ──
                from core.task.state import params_key as _params_key
                _fk = (tool_call.tool_name, _params_key(tool_call.parameters))
                state._consecutive_failures[_fk] = (
                    state._consecutive_failures.get(_fk, 0) + 1
                )
                logger.info(
                    "[dsh][task_service] approval failure count for %s: %d",
                    tool_call.tool_name,
                    state._consecutive_failures[_fk],
                )
                state.add_history(
                    f"Step {step_num}: {tool_call.tool_name}({tool_call.parameters}) "
                    f"→ FAIL: {log[:400]}"
                )

            logger.info(
                "[dsh][task_service] stream_approval: resuming orchestrator "
                "from step %d (success=%r)",
                pending.step_index + 1, result.success,
            )

            # ── 恢复 orchestrator ──
            async for evt in self._resume_orchestrator(
                pending=pending,
                state=state,
                start_step=pending.step_index + 1,
                last_tool_name=tool_call.tool_name,
                last_tool_ok=result.success,
                last_tool_log=log,
            ):
                yield evt


        finally:
            try:
                end_turn()
            except Exception:
                logger.exception("end_turn (stream_approval) 失败")
    async def _resume_orchestrator(
        self,
        *,
        pending: PendingApproval,
        state: OrchestratorState,
        start_step: int,
        last_tool_name: str,
        last_tool_ok: bool,
        last_tool_log: str,
    ) -> AsyncIterator[str]:
        task_id = pending.task_id
        session_id = pending.session_id
        turn_id = pending.turn_id or uuid.uuid4().hex[:8]
        message = pending.goal
        mode = pending.mode
        workspace = pending.workspace
        workspace_trusted = pending.workspace_trusted
        auto_discover_mcp = pending.auto_discover_mcp
        memory = self._session_provider()
        prior_messages = (
            memory.get_conversation_context(session_id, 20)
            if session_id else []
        )
        analysis_step_id = f"{task_id}:{turn_id}:analysis-resume"

        if mode == MODE_ASK:
            allowed_dangers: set = {"safe"}
            workspace_roots_for_ctx: Optional[List[str]] = None
        else:
            allowed_dangers = {"safe", "medium", "high"}
            workspace_roots_for_ctx = self._normalise_workspace(workspace) or None

        from tools.security.gateway import (
            SecurityContext, set_security_context, reset_security_context,
        )
        _sec_token = set_security_context(SecurityContext(
            allowed_danger_levels=allowed_dangers,
            workspace_roots=workspace_roots_for_ctx,
            workspace_trusted=workspace_trusted,
            session_id=session_id,
        ))

        try:
            tag = "成功" if last_tool_ok else "失败"
            tool_execution_context: List[str] = [
                f"【工具 {last_tool_name} 返回（{tag}）】\n"
                f"{(last_tool_log or '')[:8000]}"
            ]
            accumulated_resources: List[Dict[str, Any]] = []
            accumulated_resources.extend(
                self._collect_session_resources(session_id)
            )

            orchestrator = self._orchestrator_provider()
            async for event in orchestrator.run(
                message,
                allowed_danger_levels=allowed_dangers,
                allow_mcp_discovery=auto_discover_mcp,
                workspace_paths=workspace_roots_for_ctx,
                resume_state=state,
                resume_from_step=start_step,
                prior_messages=prior_messages,
                session_id=session_id,
                task_id=task_id,
            ):
                async for evt in self._process_orchestrator_event(
                    event,
                    task_id=task_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    message=message,
                    mode=mode,
                    workspace=workspace,
                    workspace_trusted=workspace_trusted,
                    auto_discover_mcp=auto_discover_mcp,
                    analysis_step_id=analysis_step_id,
                    tool_execution_context=tool_execution_context,
                    accumulated_resources=accumulated_resources,
                    event_prefix="resume",
                ):
                    yield evt
                if event.get("type") == "approval_required":
                    return

            if accumulated_resources:
                yield self._emit_step(
                    task_id,
                    self._step(
                        f"{task_id}:{turn_id}:resources",
                        f"相关资源 {len(accumulated_resources)} 个",
                        "done",
                        "resource",
                        resources=accumulated_resources,
                    ),
                )

            async for evt in self._stream_answer(
                task_id,
                session_id,
                prior_messages,
                message,
                mode,
                [],  # 无附件
                turn_id,
                rag_context=None,
                rag_resources=None,
                tool_context=tool_execution_context,
            ):
                yield evt
        finally:
            reset_security_context(_sec_token)
