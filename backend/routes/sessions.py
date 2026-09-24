"""会话管理路由。"""

from typing import Any, Callable

from fastapi import APIRouter, Form, HTTPException, Query
from fastapi.responses import FileResponse


def create_sessions_router(
    session_provider: Callable[[], Any],
) -> APIRouter:
    router = APIRouter()

    # ────────── 静态路径（必须在前） ──────────

    @router.post("/api/session/create")
    async def create_session(name: str = Form("")):
        try:
            session_id = session_provider().create_session(name)
            return {"status": "success", "session_id": session_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/api/session/list")
    async def list_sessions():
        try:
            sessions = session_provider().list_sessions()
            return {"status": "success", "sessions": sessions}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/api/session/search")
    async def search_session_messages(
        session_id: str = Query(...),
        keyword: str = Query(...),
        limit: int = Query(10, ge=1, le=100),
    ):
        try:
            results = session_provider().search_messages(session_id, keyword, limit)
            return {"status": "success", "results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/api/session/file/{file_id}/content")
    async def session_file_content(file_id: str):
        info = session_provider().get_file_by_id(file_id)
        if not info or not info.get("file_path"):
            raise HTTPException(status_code=404, detail="文件不存在")
        return FileResponse(
            path=info["file_path"],
            filename=info.get("file_name") or "file",
        )

    @router.delete("/api/session/file/{file_id}")
    async def delete_session_file(file_id: str):
        try:
            success = session_provider().delete_session_file(file_id)
            return {"status": "success", "deleted": success}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ────────── 带后续段的动态路径 ──────────

    @router.get("/api/session/{session_id}/messages")
    async def get_session_messages(session_id: str):
        try:
            messages = session_provider().get_messages(session_id)
            return {"status": "success", "messages": messages}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/api/session/{session_id}/context")
    async def get_session_context(session_id: str, max_messages: int = 20):
        try:
            messages = session_provider().get_conversation_context(
                session_id, max_messages
            )
            return {"status": "success", "messages": messages}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/session/{session_id}/message")
    async def add_session_message(
        session_id: str,
        role: str = Form(...),
        content: str = Form(...),
    ):
        try:
            msg_id = session_provider().add_message(session_id, role, content)
            return {"status": "success", "message_id": msg_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/api/session/{session_id}/files")
    async def get_session_files(session_id: str):
        try:
            files = session_provider().get_session_files(session_id)
            return {"status": "success", "files": files}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/session/{session_id}/file")
    async def add_session_file(
        session_id: str,
        file_name: str = Form(...),
        file_path: str = Form(None),
        file_type: str = Form(None),
    ):
        try:
            file_id = session_provider().add_session_file(
                session_id, file_name, file_path, file_type
            )
            return {"status": "success", "file_id": file_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/api/session/{session_id}/rename")
    async def rename_session(session_id: str, name: str = Form(...)):
        try:
            success = session_provider().rename_session(session_id, name)
            if not success:
                raise HTTPException(status_code=404, detail="会话不存在")
            return {"status": "success", "message": "会话已重命名"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ────────── 终止段动态路径（必须最后） ──────────

    @router.get("/api/session/{session_id}")
    async def get_session(session_id: str):
        try:
            info = session_provider().get_session_info(session_id)
            if info is None:
                raise HTTPException(status_code=404, detail="会话不存在")
            return {"status": "success", "session": info}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        try:
            success = session_provider().delete_session(session_id)
            if not success:
                raise HTTPException(status_code=404, detail="会话不存在")
            return {"status": "success", "message": "会话已删除"}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router