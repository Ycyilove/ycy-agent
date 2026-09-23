"""模型管理路由。"""

from typing import Any, Callable

from fastapi import APIRouter, Form, HTTPException


def create_models_router(
    model_info_provider: Callable[[], dict],
    model_downloader: Callable[[str, str | None], bool],
    model_switcher: Callable[[str], dict],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/model/info")
    async def get_model_info():
        try:
            return {"status": "success", **model_info_provider()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/model/download")
    async def download_model(
        model_name: str = Form(...),
        local_dir: str = Form(None),
    ):
        try:
            success = model_downloader(model_name, local_dir)
            if success:
                return {"status": "success", "message": "模型下载成功"}
            raise HTTPException(status_code=500, detail="模型下载失败")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/model/switch")
    async def switch_model_source(source: str = Form(...)):
        try:
            return model_switcher(source)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router