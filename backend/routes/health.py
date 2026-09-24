"""健康检查与模块状态路由。"""

from typing import Any, Callable

from fastapi import APIRouter, HTTPException


def create_health_router(
    module_status_provider: Callable[[], Any],
) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    async def root():
        return {"status": "ok", "message": "LLM API 服务运行中"}

    @router.get("/api/modules/status")
    async def modules_status():
        try:
            status = module_status_provider()
            return {"status": "success", **status}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router