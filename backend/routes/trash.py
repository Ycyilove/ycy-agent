"""Trash（回收站）HTTP 路由。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tools.tools_def.file import (
    list_trash,
    restore_from_trash,
    purge_from_trash,
    purge_all_trash,
)

logger = logging.getLogger(__name__)


class TrashIdBody(BaseModel):
    id: str
    session_id: Optional[str] = None

def create_trash_router() -> APIRouter:
    router = APIRouter(prefix="/api/trash", tags=["trash"])

    @router.get("/list")
    def _list() -> Dict[str, Any]:
        try:
            items = list_trash()
            return {"items": items, "count": len(items)}
        except Exception as e:
            logger.exception("list_trash failed")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/restore")
    def _restore(body: TrashIdBody) -> Dict[str, Any]:
        result = restore_from_trash(body.id)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        # ── 写 file_log ──
        if body.session_id:
            try:
                from backend.services.session_memory import get_session_memory
                get_session_memory().add_file_log(body.session_id, {
                    "id": f"restore-{body.id}-{int(time.time() * 1000)}",
                    "action": "restore",
                    "path": result.get("restored_path") or result.get("original_path") or "",
                    "time": int(time.time() * 1000),
                    "status": "success",
                })
            except Exception:
                logger.exception("记录 restore file_log 失败")
        
        return result


    @router.post("/purge")
    def _purge(body: TrashIdBody) -> Dict[str, Any]:
        result = purge_from_trash(body.id)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        # ── 写 file_log ──
        if body.session_id:
            try:
                from backend.services.session_memory import get_session_memory
                get_session_memory().add_file_log(body.session_id, {
                    "id": f"purge-{body.id}-{int(time.time() * 1000)}",
                    "action": "delete",
                    "path": result.get("path") or "",
                    "time": int(time.time() * 1000),
                    "status": "success",
                })
            except Exception:
                logger.exception("记录 purge file_log 失败")
        
        return result

    @router.post("/purge_all")
    def _purge_all() -> Dict[str, Any]:
        return purge_all_trash()

    return router