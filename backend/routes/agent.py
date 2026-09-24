"""Agent workbench task and approval routes."""

import json
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..services.task_service import with_sse_done


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
        message: str = Form(''),
        history: str = Form('[]'),
        mode: str = Form('plan'),
        workspace: str = Form('[]'),
        session_id: Optional[str] = Form(None),
        task_id: Optional[str] = Form(None),
        auto_discover_mcp: str = Form('false'),
        files: Optional[List[UploadFile]] = File(None),
    ):
        if not message.strip() and not files:
            raise HTTPException(status_code=400, detail='任务内容和附件不能同时为空')

        parsed_history = _parse_json_list(history, 'history')
        parsed_workspace = _parse_json_list(workspace, 'workspace')
        parsed_auto_discover = _parse_bool(auto_discover_mcp, default=False)

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
            workspace=[str(path) for path in parsed_workspace],
            session_id=session_id,
            auto_discover_mcp=parsed_auto_discover,
            attachments=attachments,
        )
        return StreamingResponse(
            with_sse_done(stream),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            },
        )

    @router.post('/api/agent/tasks/{task_id}/approve')
    async def approve_agent_task(
        task_id: str,
        approval_id: str = Form(...),
        action: str = Form('allow'),
    ):
        stream = task_service_provider().stream_approval(task_id, approval_id, action)
        return StreamingResponse(
            with_sse_done(stream),
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

    @router.get('/api/agent/tasks/{task_id}')
    async def get_agent_task(task_id: str):
        task = session_provider().get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail='任务不存在')
        return {'status': 'success', 'task': task}

    @router.delete('/api/agent/tasks/{task_id}')
    async def delete_agent_task(task_id: str):
        deleted = session_provider().delete_task(task_id)
        if not deleted:
            raise HTTPException(status_code=404, detail='任务不存在')
        return {'status': 'success', 'deleted': True}

    return router