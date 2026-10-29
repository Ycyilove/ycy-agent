"""Agent workbench task and approval routes."""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from core.task.service import with_sse_done
from core.task.modes import MODE_AGENT, validate_mode

logger = logging.getLogger(__name__)


def _resolve_workspace_paths(paths: List[str]) -> List[str]:
    """解析客户端传的 workspace 路径（用户指定模式）。

    规则：
        1. 必须是非空绝对路径（Windows: C:/... 或 /...；Unix: /...）
        2. 相对路径 → 400
        3. 不做白名单校验——安全由 SecurityGateway 的写操作审批兜底
    """
    resolved: List[str] = []
    for raw in paths or []:
        if not isinstance(raw, str) or not raw.strip():
            continue
        raw = raw.strip()

        if not os.path.isabs(raw):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"workspace 必须是绝对路径（如 D:/projects/foo）: {raw}"
                ),
            )

        candidate = os.path.abspath(raw)
        if not os.path.exists(candidate):
            logger.warning(
                "[workspace] 路径不存在，仍接受（用户可能稍后创建）: %s",
                candidate,
            )
        resolved.append(candidate)

    return resolved


def _parse_json_list(value: str, field_name: str) -> List[Any]:
    try:
        parsed = json.loads(value or '[]')
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail=f'{field_name} 格式无效') from error
    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail=f'{field_name} 必须是数组')
    return parsed


def _parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def create_agent_router(
    task_service_provider: Callable[[], Any],
    session_provider: Callable[[], Any],
) -> APIRouter:
    router = APIRouter()

    @router.post('/api/agent/tasks/stream')
    async def stream_agent_task(
        request: Request,
        message: str = Form(''),
        history: str = Form('[]'),
        mode: str = Form(MODE_AGENT),
        workspace: str = Form('[]'),
        session_id: Optional[str] = Form(None),
        task_id: Optional[str] = Form(None),
        auto_discover_mcp: str = Form('false'),
        files: Optional[List[UploadFile]] = File(None),
    ):
        try:
            mode = validate_mode(mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if not message.strip() and not files:
            raise HTTPException(status_code=400, detail='任务内容和附件不能同时为空')

        parsed_history = _parse_json_list(history, 'history')
        parsed_workspace_raw = _parse_json_list(workspace, 'workspace')
        parsed_auto_discover = _parse_bool(auto_discover_mcp, default=False)

        # ── workspace：用户在 UI 选的绝对路径，原样接受 ──
        parsed_workspace = _resolve_workspace_paths(
            [str(p) for p in parsed_workspace_raw]
        )

        attachments = []
        for upload in files or []:
            attachments.append({
                'filename': upload.filename or 'attachment',
                'content_type': upload.content_type,
                'content': await upload.read(),
            })

        stream = task_service_provider().stream_task(
            task_id=task_id,
            message=message,
            history=parsed_history,
            mode=mode,
            workspace=parsed_workspace,
            session_id=session_id,
            auto_discover_mcp=parsed_auto_discover,
            attachments=attachments,
        )
        return StreamingResponse(
            with_sse_done(stream, is_disconnected=request.is_disconnected),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            },
        )

    @router.post('/api/agent/tasks/{task_id}/approve')
    async def approve_agent_task(
        request: Request,
        task_id: str,
        approval_id: str = Form(...),
        action: str = Form('deny'),
        session_id: str = Form(...),
    ):
        if action not in ('allow', 'deny', 'allow-always'):
            raise HTTPException(status_code=400, detail='action 非法')

        # 校验 task 归属
        task = session_provider().get_task(task_id)
        if not task or task.get('session_id') != session_id:
            raise HTTPException(status_code=404, detail='任务不存在')

        # 服务端权威审批状态
        from tools.security.approval import get_approval_store
        store = get_approval_store()
        consumed = store.decide(approval_id, session_id, action)
        if not consumed:
            # 已消费 / 过期 / session 不符 → 视为 stale，走原 stream_approval 的 stale 分支
            logger.warning(
                "[approve] approval %s 已失效或不属于 session %s",
                approval_id, session_id,
            )

        stream = task_service_provider().stream_approval(
            task_id, approval_id, action
        )
        return StreamingResponse(
            with_sse_done(stream, is_disconnected=request.is_disconnected),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            },
        )
        
    @router.get('/api/agent/tasks')
    async def list_agent_tasks(session_id: str):
        return {'status': 'success', 'tasks': session_provider().list_tasks(session_id)}

    @router.delete('/api/agent/tasks/{task_id}')
    async def delete_agent_task(task_id: str, session_id: str):
        # ── 安全：校验 task 归属 ──
        task = session_provider().get_task(task_id)
        if not task or task.get('session_id') != session_id:
            raise HTTPException(status_code=404, detail='任务不存在')
        deleted = session_provider().delete_task(task_id)
        if not deleted:
            raise HTTPException(status_code=404, detail='任务不存在')
        return {'status': 'success', 'deleted': True}

    return router