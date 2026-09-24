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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from tools.agent import ToolAgent, ToolCallRequest, ToolCallResult

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

    @staticmethod
    def _file_log_entry(
        tool_call: ToolCallRequest, result: ToolCallResult
    ) -> Optional[Dict[str, Any]]:
        action_map = {
            "list_files": "read",
            "read_csv": "read",
            "read_excel": "read",
            "get_file_info": "read",
            "rename_file": "move",
            "convert_file_format": "write",
        }
        action = action_map.get(tool_call.tool_name)
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

    def _extract_resources_from_tool_result(
        self, result: ToolCallResult
    ) -> List[Dict[str, Any]]:
        if not result or not getattr(result, "success", False):
            return []
        payload = result.result
        if not isinstance(payload, dict):
            return []

        resources: List[Dict[str, Any]] = []

        listed = payload.get("resources")
        if isinstance(listed, list):
            for item in listed:
                if isinstance(item, dict) and item.get("url"):
                    resources.append(item)

        single = payload.get("resource")
        if isinstance(single, dict) and single.get("url"):
            resources.append(single)

        columns = payload.get("columns")
        rows = payload.get("rows")
        if isinstance(columns, list) and isinstance(rows, list) and columns:
            resources.append({
                "id": f"table-{uuid.uuid4().hex[:8]}",
                "kind": "table",
                "filename": payload.get("filename") or result.tool_name,
                "columns": columns,
                "rows": rows,
            })

        return resources

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
            provider = self._llm_provider().stream_generate(messages, mode="quick")

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

        running_step = self._step(
            step_id, label, "running", "action", path=self._tool_path(tool_call)
        )
        yield self._emit_step(task_id, running_step)

        result = await self._agent_provider().execute_tool_async(tool_call)
        log = self._agent_provider().format_tool_result(result)
        status = "done" if result.success else "failed"

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
            yield sse_event("file_log", entry=file_log)

        self._session_provider().add_message(session_id, "assistant", log)
        final_status = "done" if result.success else "failed"
        self._persist_task_status(task_id, final_status)
        yield sse_event("done", task_status=final_status)

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

        # 打开自动发现时，重置本任务已试过的 MCP server 集合；
        # 关闭时不做任何事（orchestrator 会因为 allowed_tool_names 里没有
        # search_and_connect_mcp 而根本不会触发 discovery）。
        if auto_discover_mcp:
            try:
                from tools.mcp_discovery import reset_tried_servers
                reset_tried_servers()
            except Exception:
                logger.exception("重置 MCP discovery 状态失败")
                
        memory = self._session_provider()
        prior_messages = (
            memory.get_conversation_context(session_id, 20)
            if session_id
            else history
        )
        if not prior_messages:
            prior_messages = history
        self._record_session_input(session_id, message, attachments)

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
            ):
                event_type = event.get("type")

                if event_type == "thinking":
                    thought = (event.get("thought") or "").strip()
                    if thought:
                        yield self._emit_step(
                            task_id,
                            self._step(
                                f"{task_id}:{turn_id}:orch-think:{event['step']}",
                                thought[:120],
                                "done",
                                "think",
                            ),
                        )

                elif event_type == "tool_start":
                    tool_call = event["tool_call"]
                    yield self._emit_step(
                        task_id,
                        self._step(
                            f"{task_id}:{turn_id}:orch-tool:{event['step']}",
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
                    yield self._emit_step(
                        task_id,
                        self._step(
                            f"{task_id}:{turn_id}:orch-tool:{event['step']}",
                            label,
                            status,
                            "action",
                            path=self._tool_path(tool_call),
                            log=event.get("log"),
                        ),
                    )

                    if result.result is not None:
                        try:
                            if isinstance(result.result, dict):
                                payload = json.dumps(
                                    result.result, ensure_ascii=False
                                )
                            else:
                                payload = str(result.result)
                            status_tag = "成功" if result.success else "失败"
                            error_note = (
                                f"\n错误: {result.error}"
                                if not result.success and result.error
                                else ""
                            )
                            tool_execution_context.append(
                                f"【工具 {tool_call.tool_name} 返回（{status_tag}）】"
                                f"{error_note}\n{payload[:8000]}"
                            )
                        except Exception:
                            logger.exception("累积工具结果失败")

                    try:
                        found = self._extract_resources_from_tool_result(result)
                        if found:
                            accumulated_resources.extend(found)
                    except Exception:
                        logger.exception("提取工具资源失败")

                    file_log = self._file_log_entry(tool_call, result)
                    if file_log:
                        yield sse_event("file_log", entry=file_log)

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
                    with self._lock:
                        self._pending[approval_id] = pending

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

    async def stream_approval(
        self,
        task_id: str,
        approval_id: str,
        action: str,
    ) -> AsyncIterator[str]:
        with self._lock:
            pending = self._pending.pop(approval_id, None)
        if pending is None or pending.task_id != task_id:
            yield sse_event("error", message="审批已失效或不存在")
            yield sse_event("done", task_status="failed")
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

        async for event in self._execute_tool(
            pending.task_id,
            pending.session_id,
            pending.tool_call,
            pending.turn_id or "",
            pending.step_id,
        ):
            yield event