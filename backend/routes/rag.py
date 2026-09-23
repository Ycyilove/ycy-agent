"""RAG 路由：legacy + 本地持久化 RAG。"""

from typing import Any, Callable, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


def create_rag_router(
    legacy_rag_provider: Callable[[], Any],
    local_rag_provider: Callable[[], Any],
    llm_provider: Callable[[], Any],
) -> APIRouter:
    router = APIRouter()

    # ────────── Legacy RAG ──────────

    @router.post("/api/rag/add")
    def rag_add_texts(texts: List[str] = Form(...)):
        try:
            service = legacy_rag_provider()
            result = service.add_texts(texts)
            return {"status": "success", "message": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/rag/query")
    def rag_query(query: str = Form(...)):
        try:
            service = legacy_rag_provider()
            result = service.rag_answer(query, llm_provider())
            return {"status": "success", "result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ────────── Local RAG ──────────

    @router.post("/api/rag/add_file")
    async def rag_add_file(file: UploadFile = File(...)):
        try:
            file_content = await file.read()
            result = local_rag_provider().add_file(file_content, file.filename)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/rag/add_files")
    async def rag_add_files(files: List[UploadFile] = File(...)):
        try:
            file_list = []
            for file in files:
                file_content = await file.read()
                file_list.append((file_content, file.filename))
            result = local_rag_provider().add_files(file_list)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/api/rag/stats")
    async def rag_stats():
        try:
            stats = local_rag_provider().get_stats()
            return {"status": "success", **stats}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/rag/search")
    async def rag_search(query: str = Form(...), top_k: int = Form(3)):
        try:
            results = local_rag_provider().search(query, top_k)
            return {"status": "success", "results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/api/rag/resource/{resource_id}")
    async def rag_resource(resource_id: str):
        resource_manager = local_rag_provider().resource_manager
        resource = resource_manager.path_for(resource_id)
        if resource is None:
            raise HTTPException(status_code=404, detail="资源不存在")
        return FileResponse(
            path=str(resource_manager.file_path(resource)),
            media_type=resource.mime_type,
            filename=resource.filename,
        )

    @router.post("/api/rag/query/stream")
    async def rag_query_stream(query: str = Form(...), top_k: int = Form(3)):
        try:
            return StreamingResponse(
                local_rag_provider().stream_rag_answer(query, top_k),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/api/rag/file")
    async def rag_delete_file(filename: str = Form(...)):
        try:
            return local_rag_provider().delete_file(filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/api/rag/clear")
    async def rag_clear_all():
        try:
            return local_rag_provider().clear_all()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router