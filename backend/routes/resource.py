"""统一资源下载路由。"""

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ..services.resource_store import get, get_path

logger = logging.getLogger(__name__)


def create_resource_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/resources/{resource_id}")
    async def download_resource(resource_id: str):
        res = get(resource_id)
        if res is None:
            raise HTTPException(status_code=404, detail="资源不存在或已过期")

        # 表格资源：直接返回 JSON
        if res.kind == "table":
            return JSONResponse({
                "kind": "table",
                "columns": res.columns,
                "rows": res.rows,
                "filename": res.filename,
            })

        path = get_path(resource_id)
        if path is None:
            raise HTTPException(status_code=404, detail="资源文件已丢失")

        inline_ok = (
            res.mime.startswith("image/")
            or res.mime == "application/pdf"
            or res.mime.startswith("text/")
            or res.mime.startswith("audio/")
            or res.mime.startswith("video/")
        )
        return FileResponse(
            path,
            media_type=res.mime,
            filename=res.filename,
            content_disposition_type="inline" if inline_ok else "attachment",
        )

    return router