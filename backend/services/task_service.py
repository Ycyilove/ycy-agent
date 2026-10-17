"""Task orchestration for the Agent workbench.

关键设计：
    - RAG 兜底已移除——让 orchestrator 通过 search_knowledge 工具自己决策
    - thinking/answer 流解析抽到 stream_parser
    - prompt 构造抽到 prompts
    - 保留了 SSE 心跳、审批流、持久化等机制
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from tools.agent import ToolAgent, ToolCallRequest, ToolCallResult
from tools.orchestration.state import OrchestratorState

from .prompts import (
    attachment_context as build_attachment_context,
    attachment_label,
    build_prompt,
)
from .stream_parser import (
    StreamSplitter,
    async_iter_sync as _async_iter_sync,
    parse_provider_chunk,
)

from .llm_debug_log import begin_turn, end_turn
from .observability import set_turn_context

logger = logging.getLogger(__name__)


MODES = {"ask", "plan", "craft"}
WORKSPACE_PATH_KEYS = {
    "directory",
    "file_path",
    "old_path",
    "output_path",
    "path",
}


@dataclass
class PendingApproval:
    task_id: str
    session_id: str
    step_id: str
    approval_id: str
    tool_call: ToolCallRequest
    title: str
    turn_id: str = ""
    path: Optional[str] = None

    # ── orchestrator 恢复所需信息 ──
    goal: str = ""
    mode: str = "plan"
    workspace: Optional[List[str]] = None
    auto_discover_mcp: bool = False
    step_index: int = 1
    state_snapshot: Optional[dict] = None

    # ── 内部：创建时间，用于 TTL 清理 ──
    _created_at: float = field(default_factory=time.time)

def _looks_base64(s: str) -> bool:
    if not isinstance(s, str) or len(s) < 32:
        return False
    import re as _re
    return bool(_re.match(r"^[A-Za-z0-9+/=\s]+$", s[:256]))


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


async def with_sse_done(events: AsyncIterator[str]) -> AsyncIterator[str]:
    """Append the conventional SSE terminator, with periodic heartbeat."""
    HEARTBEAT_INTERVAL = 15.0

    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _DONE = object()

    async def pump():
        try:
            async for event in events:
                await queue.put(event)
        except Exception:
            logger.exception("[dsh][sse] upstream generator raised")
            raise
        finally:
            await queue.put(_DONE)

    pump_task = asyncio.create_task(pump())

    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=HEARTBEAT_INTERVAL
                )
            except asyncio.TimeoutError:
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

    yield "data: [DONE]\n\n"


class AgentTaskService:
    """Coordinate task planning, tool execution, approvals, and persistence."""

    def __init__(
        self,
        agent_provider: Callable[[], ToolAgent],
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
            logger.info("[dsh][pending] purged %d expired approvals", len(expired))

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

    @staticmethod
    def _path_in_workspace(path: str, workspace: List[str]) -> bool:
        if not workspace:
            return True
        absolute = os.path.abspath(path)
        return any(
            absolute == root or absolute.startswith(root + os.sep)
            for root in workspace
        )

    def _needs_workspace_confirmation(
        self,
        tool_call: ToolCallRequest,
        workspace: List[str],
    ) -> bool:
        if not workspace:
            return False
        for key, value in tool_call.parameters.items():
            if key not in WORKSPACE_PATH_KEYS or not isinstance(value, str):
                continue
            if value and not self._path_in_workspace(value, workspace):
                return True
        return False

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

        避免 json.dumps / prompt 里出现原始二进制。
        - dict / list / tuple 递归处理
        - bytes 替换成 "<binary N bytes>"
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

        from .resource_store import (
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
        answer_step_id = f"{task_id}:{turn_id}:answer"
        thinking_step_id = f"{task_id}:{turn_id}:thinking"

        initial_step = self._step(
            thinking_step_id, "思考过程", "running", "think", log=""
        )
        yield self._emit_step(task_id, initial_step)

        # 隐式兜底：把检索到的资源作为时间线步骤展示
        if rag_resources:
            resource_step = self._step(
                f"{task_id}:{turn_id}:rag_resources",
                f"引用知识库 {len(rag_resources)} 个资源",
                "done",
                "resource",
                resources=rag_resources,
            )
            yield self._emit_step(task_id, resource_step)

        # 选择 provider
        if attachments and all(self._is_multimodal(item) for item in attachments):
            provider = self._multimodal_provider().stream_analyze(
                self._build_multimodal_message(message, attachments)
            )
        else:
            prompt = build_prompt(
                message,
                mode,
                attachments,
                rag_context,
                tool_context=tool_context,
            )
            messages = list(prior_messages) + [{"role": "user", "content": prompt}]
            provider = self._llm_provider().stream_generate(messages, mode="deep")

            # 落盘 stream_answer 的输入
            try:
                from .llm_debug_log import dump_llm_call
                full_prompt = "\n\n".join(
                    f"[{m.get('role', 'user')}]\n{m.get('content', '')}"
                    for m in messages
                )
                dump_llm_call(
                    kind="stream_answer",
                    prompt=full_prompt,
                    response=None,   # 流式响应在结束时才拿到，下面单独落盘
                    model=None,
                    extra={"mode": mode, "message": message[:200]},
                )
            except Exception:
                logger.exception("dump stream_answer prompt failed")

        # 用 StreamSplitter 解析
        splitter_events: List[Dict[str, Any]] = []

        def on_thinking(text: str) -> None:
            step = self._step(
                thinking_step_id, "思考过程", "running", "think", log=text
            )
            splitter_events.append({"kind": "thinking", "step": step})

        def on_answer(text: str) -> None:
            step = self._step(
                answer_step_id, "正在生成回答…", "running", "answer", log=text
            )
            splitter_events.append({"kind": "answer", "step": step})

        splitter = StreamSplitter(
            on_thinking=on_thinking, on_answer=on_answer
        )

        async for chunk in _async_iter_sync(provider):
            for payload in parse_provider_chunk(chunk):
                result = splitter.feed(payload)
                if result == "error":
                    yield sse_event(
                        "error",
                        message=payload.get("text") or "生成回答失败",
                    )

                # 把本轮 splitter 产生的事件 yield 出去
                while splitter_events:
                    evt = splitter_events.pop(0)
                    yield self._emit_step(task_id, evt["step"])

        # finalize
        thinking_text, answer_text = splitter.finalize()

        # finalize 可能再次触发 on_thinking/on_answer
        while splitter_events:
            evt = splitter_events.pop(0)
            yield self._emit_step(task_id, evt["step"])

        final_think = self._step(
            thinking_step_id, "思考过程", "done", "think", log=thinking_text
        )
        yield self._emit_step(task_id, final_think)

        if not answer_text:
            answer_text = "未生成有效回答。"

        # [[asset:xxx]] 解析
        if rag_resources:
            try:
                rag = self._rag_provider()
                answer_text = rag.resource_manager.resolve_asset_tags(
                    answer_text, rag_resources
                )
            except Exception as error:
                logger.warning("解析 asset 标签失败: %s", error)

        final_answer = self._step(
            answer_step_id, "Agent 回复", "done", "answer", log=answer_text
        )
        yield self._emit_step(task_id, final_answer)

        # 落盘 stream_answer 的响应
        try:
            from .llm_debug_log import dump_llm_call
            dump_llm_call(
                kind="stream_answer_response",
                prompt="",                      # prompt 已经落过，不重复
                response=answer_text,
                extra={
                    "thinking_len": len(thinking_text or ""),
                    "answer_len": len(answer_text or ""),
                },
            )
        except Exception:
            logger.exception("dump stream_answer response failed")

        self._session_provider().add_message(session_id, "assistant", answer_text)
        self._persist_task_status(task_id, "done")
        yield sse_event("done", task_status="done")

    # ────────── 工具执行（回退路径） ──────────

    async def _execute_tool(
        self,
        task_id: str,
        session_id: str,
        tool_call: ToolCallRequest,
        turn_id: str,
        step_id: Optional[str] = None,
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
            approval_id = f"approval-{uuid.uuid4().hex}"
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
            }
            waiting_step = self._step(
                tool_step_id,
                title,
                "waiting",
                "action",
                path=pending.path,
                approval=approval,
            )
            yield self._emit_step(
                task_id,
                self._step(
                    analysis_step_id, "任务分析完成", "done", "think"
                ),
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
                    accumulated_resources.extend(found)
            except Exception:
                logger.exception("提取工具资源失败")

            # ── 2. 剥离 bytes（防止污染 json） ──
            if isinstance(getattr(result, "result", None), dict):
                result.result = self._strip_bytes(result.result)

            log_parts: List[str] = []
            if event.get("log"):
                log_parts.append(str(event["log"]))

            result_payload = getattr(result, "result", None)
            if result_payload is not None:
                try:
                    if isinstance(result_payload, (dict, list, tuple)):
                        payload_text = json.dumps(
                            result_payload, ensure_ascii=False, indent=2,
                            default=str,     # ← bytes 转成字符串
                        )
                    else:
                        payload_text = str(result_payload)
                    if len(payload_text) > 4000:
                        payload_text = payload_text[:4000] + "\n… (truncated)"
                    log_parts.append("【原始返回】\n" + payload_text)
                except Exception:
                    logger.exception("拼接工具原始返回失败")

            if not result.success and getattr(result, "error", None):
                log_parts.append(f"【错误】{result.error}")

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
                    if isinstance(result.result, dict):
                        payload = json.dumps(
                            result.result, ensure_ascii=False,
                            default=str,     # ← bytes 转成字符串
                        )
                    else:
                        payload = str(result.result)
                    tag = "成功" if result.success else "失败"
                    err_note = (
                        f"\n错误: {result.error}"
                        if not result.success and result.error
                        else ""
                    )
                    tool_execution_context.append(
                        f"【工具 {tool_call.tool_name} 返回（{tag}）】"
                        f"{err_note}\n{payload[:8000]}"
                    )
                except Exception:
                    logger.exception("累积工具结果失败")

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

        elif event_type == "done":
            yield self._emit_step(
                task_id,
                self._step(
                    analysis_step_id, "任务分析完成", "done", "think"
                ),
            )

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
        mode = mode if mode in MODES else "plan"
        session_id = self._ensure_session(session_id, message)

        # ── 每个任务开头重置 per-task 状态 ──
        # contextvars 在新 asyncio task 里是继承的副本，但显式清一次更保险，
        # 避免把上一个任务的允许工具集带进来。
        try:
            self._agent_provider().clear_available_tools()
        except Exception:
            logger.exception("clear_available_tools failed")

        # ── 开启 turn：把本次提问的所有 LLM 调用归到同一目录 ──
        begin_turn(
            session_id=session_id,
            task_id=task_id,
            question=message,
        )

        # ── 开启 mem0 记录（per-task） ──
        try:
            from .mem0_debug_log import begin_task as begin_mem0_task
            begin_mem0_task(session_id=session_id, task_id=task_id)
        except Exception:
            logger.exception("begin_mem0_task failed")

        with set_turn_context(
            session_id=session_id,
            task_id=task_id,
            question=message,
        ):
          try:
            # 说明：tried_servers 的重置由 orchestrator.run() 内部完成，
            # 这里是 contextvars 语义，per-task 隔离，不需要重复 reset。
                    
            memory = self._session_provider()
            prior_messages = (
                memory.get_conversation_context(session_id, 20)
                if session_id
                else history
            )
            if not prior_messages:
                prior_messages = history
            self._record_session_input(session_id, message, attachments)

            # ── Mem0：从用户消息提取路径 ──
            try:
                from .memory_layer import remember_user_message
                remember_user_message(session_id, message)
            except Exception:
                logger.exception("remember_user_message 失败")

            self._persist_task(
                task_id=task_id,
                session_id=session_id,
                message=message,
                mode=mode,
                workspace=workspace,
                attachments=attachments,
            )

            yield sse_event(
                "task", task_id=task_id, session_id=session_id, turn_id=turn_id
            )

            if message.strip():
                yield self._emit_step(
                    task_id,
                    self._step(
                        f"{task_id}:{turn_id}:user",
                        message.strip(),
                        "done",
                        "user",
                    ),
                )

            for index, attachment in enumerate(attachments):
                step = self._step(
                    f"{task_id}:{turn_id}:attachment:{index}",
                    f"接收附件：{attachment_label(attachment)}",
                    "done",
                    "action",
                    path=attachment_label(attachment),
                )
                yield self._emit_step(task_id, step)

            analysis_step_id = f"{task_id}:{turn_id}:analysis"
            yield self._emit_step(
                task_id,
                self._step(analysis_step_id, "正在理解任务…", "running", "think"),
            )

            if mode == "ask":
                allowed_dangers: Optional[set] = {"safe"}
            else:
                allowed_dangers = {"safe", "medium", "high"}

            accumulated_resources: List[Dict[str, Any]] = []
            accumulated_resources.extend(self._collect_session_resources(session_id))
            tool_execution_context: List[str] = []

            used_orchestrator = False
            if self._orchestrator_provider is not None:
                used_orchestrator = True
                orchestrator = self._orchestrator_provider()
                async for event in orchestrator.run(
                    message,
                    allowed_danger_levels=allowed_dangers,
                    allow_mcp_discovery=auto_discover_mcp,
                    workspace_paths=self._normalise_workspace(workspace) if mode == "plan" else None,
                    plan_mode=(mode == "plan"),
                    prior_messages=prior_messages,
                    session_id=session_id,
                ):
                    async for evt in self._process_orchestrator_event(
                        event,
                        task_id=task_id,
                        session_id=session_id,
                        turn_id=turn_id,
                        message=message,
                        mode=mode,
                        workspace=workspace,
                        auto_discover_mcp=auto_discover_mcp,
                        analysis_step_id=analysis_step_id,
                        tool_execution_context=tool_execution_context,
                        accumulated_resources=accumulated_resources,
                        event_prefix="orch",
                    ):
                        yield evt
                    if event.get("type") == "approval_required":
                        return

            # 无 orchestrator 时的回退路径
            if not used_orchestrator:
                tool_call: Optional[ToolCallRequest] = None
                if mode != "ask":
                    try:
                        attachment_names = ", ".join(
                            attachment_label(item) for item in attachments
                        )
                        intent_message = message
                        if attachment_names:
                            intent_message += f"\n当前附件：{attachment_names}"
                        intent = self._agent_provider().analyze_intent(
                            intent_message, use_llm=True
                        )
                        tool_call = self._agent_provider().prepare_tool_call(intent)
                    except Exception as error:
                        yield sse_event("error", message=f"任务分析失败：{error}")

                if tool_call:
                    workspace_paths = self._normalise_workspace(workspace)
                    needs_confirmation = (
                        mode == "plan"
                        or tool_call.user_confirmation_needed
                        or self._needs_workspace_confirmation(tool_call, workspace_paths)
                    )
                    yield self._emit_step(
                        task_id,
                        self._step(
                            analysis_step_id,
                            f"已识别操作：{tool_call.tool_name}",
                            "done",
                            "think",
                        ),
                    )
                    if needs_confirmation:
                        approval_id = f"approval-{uuid.uuid4().hex}"
                        tool_step_id = f"{task_id}:{turn_id}:approval:{approval_id}"
                        title = f"确认执行工具：{tool_call.tool_name}"
                        pending = PendingApproval(
                            task_id=task_id,
                            session_id=session_id,
                            step_id=tool_step_id,
                            approval_id=approval_id,
                            tool_call=tool_call,
                            title=title,
                            turn_id=turn_id,
                            path=self._tool_path(tool_call),
                        )
                        self._store_pending(pending)

                        approval = {
                            "id": approval_id,
                            "title": title,
                            "toolName": tool_call.tool_name,
                            "parameters": tool_call.parameters,
                            "dangerLevel": "high"
                            if tool_call.user_confirmation_needed
                            else "medium",
                            "path": pending.path,
                            "taskId": task_id,
                            "stepId": tool_step_id,
                        }
                        waiting_step = self._step(
                            tool_step_id,
                            title,
                            "waiting",
                            "action",
                            path=pending.path,
                            approval=approval,
                        )
                        yield self._emit_step(task_id, waiting_step)
                        self._persist_task_status(task_id, "waiting")
                        yield sse_event("done", task_status="waiting")
                        return

                    async for event in self._execute_tool(
                        task_id, session_id, tool_call, turn_id
                    ):
                        yield event

                    try:
                        fallback_result = await self._agent_provider().execute_tool_async(
                            tool_call
                        )
                        if fallback_result.success and fallback_result.result is not None:
                            if isinstance(fallback_result.result, dict):
                                payload = json.dumps(
                                    fallback_result.result, ensure_ascii=False
                                )
                            else:
                                payload = str(fallback_result.result)
                            tool_execution_context.append(
                                f"【工具 {tool_call.tool_name} 返回】\n{payload[:8000]}"
                            )
                    except Exception:
                        logger.exception("回退路径累积工具结果失败")

                    return

                yield self._emit_step(
                    task_id,
                    self._step(
                        analysis_step_id, "已完成任务分析", "done", "think"
                    ),
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
            rag_resources: Optional[List[Dict[str, Any]]] = None

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
            end_turn()
            try:
                from .mem0_debug_log import end_task as end_mem0_task
                end_mem0_task()
            except Exception:
                logger.exception("end_mem0_task failed")

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
            logger.info(
                "[dsh][task_service] stream_approval: id 已失效，静默忽略"
            )
            yield sse_event("done", task_status="stale")
            return

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

        try:
            result = await self._agent_provider().execute_tool_async(tool_call)
        except Exception as error:
            logger.exception(
                "[dsh][task_service] approval execution raised: %s",
                tool_call.tool_name,
            )
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
                session_id=session_id,
            )
            if found:
                accumulated_resources.extend(found)
        except Exception:
            logger.exception("提取工具资源失败")

        # ── 再剥离 bytes（防止污染 context / json） ──
        if isinstance(getattr(result, "result", None), dict):
            result.result = self._strip_bytes(result.result)

        log_parts: List[str] = []
        if event.get("log"):
            log_parts.append(str(event["log"]))

        result_payload = getattr(result, "result", None)
        if result_payload is not None:
            try:
                if isinstance(result_payload, (dict, list, tuple)):
                    payload_text = json.dumps(
                        result_payload, ensure_ascii=False, indent=2
                    )
                else:
                    payload_text = str(result_payload)
                if len(payload_text) > 4000:
                    payload_text = payload_text[:4000] + "\n… (truncated)"
                log_parts.append("【原始返回】\n" + payload_text)
            except Exception:
                logger.exception("拼接工具原始返回失败")
        if not result.success and getattr(result, "error", None):
            log_parts.append(f"【错误】{result.error}")
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

        # ── 提取二进制资源 ──
        try:
            found = self._extract_resources_from_tool_result(
                result,
                task_id=task_id,
                session_id=pending.session_id,
            )
            if found:
                yield self._emit_step(
                    task_id,
                    self._step(
                        f"{task_id}:{step_id}:resources",
                        f"工具产生 {len(found)} 个资源",
                        "done",
                        "resource",
                        resources=found,
                    ),
                )
        except Exception:
            logger.exception("提取工具资源失败")

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

        if result.success:
            state.mark_success(tool_call.tool_name, tool_call.parameters, step_num)
            state.add_history(
                f"Step {step_num}: {tool_call.tool_name}({tool_call.parameters}) "
                f"→ OK: {log[:400]}"
            )

            # ── Mem0：审批后成功也写事实 ──
            try:
                from .memory_layer import remember_tool_success
                remember_tool_success(
                    session_id=state.session_id or pending.session_id,
                    tool_name=tool_call.tool_name,
                    params=tool_call.parameters,
                    result=result.result,
                )
            except Exception:
                logger.exception("remember_tool_success (approval) 失败")
        else:
            state.mark_failure(
                tool_call.tool_name, tool_call.parameters, result.error or ""
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
        """从审批后的状态恢复 orchestrator，继续循环。

        自动注入最后一次工具执行的结果，让 LLM 决定下一步。
        """
        task_id = pending.task_id
        session_id = pending.session_id
        turn_id = pending.turn_id or uuid.uuid4().hex[:8]
        message = pending.goal
        mode = pending.mode
        workspace = pending.workspace
        auto_discover_mcp = pending.auto_discover_mcp
        memory = self._session_provider()
        prior_messages = (
            memory.get_conversation_context(session_id, 20)
            if session_id
            else []
        )
        
        analysis_step_id = f"{task_id}:{turn_id}:analysis-resume"

        if mode == "ask":
            allowed_dangers: Optional[set] = {"safe"}
        else:
            allowed_dangers = {"safe", "medium", "high"}

        # 把最后一次工具结果作为上下文传入
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
            workspace_paths=(
                self._normalise_workspace(workspace)
                if mode == "plan"
                else None
            ),
            plan_mode=(mode == "plan"),
            resume_state=state,
            resume_from_step=start_step,
            prior_messages=prior_messages,
            session_id=session_id,
        ):
            async for evt in self._process_orchestrator_event(
                event,
                task_id=task_id,
                session_id=session_id,
                turn_id=turn_id,
                message=message,
                mode=mode,
                workspace=workspace,
                auto_discover_mcp=auto_discover_mcp,
                analysis_step_id=analysis_step_id,
                tool_execution_context=tool_execution_context,
                accumulated_resources=accumulated_resources,
                event_prefix="resume",
            ):
                yield evt
            if event.get("type") == "approval_required":
                return

        # orchestrator 正常结束（done / max_steps / error）
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

        memory = self._session_provider()
        prior_messages = (
            memory.get_conversation_context(session_id, 20)
            if session_id
            else []
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