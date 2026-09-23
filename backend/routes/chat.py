"""对话与多模态路由。"""

import base64
import json
from typing import Any, Callable, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


def create_chat_router(
    llm_provider: Callable[[], Any],
    multimodal_provider: Callable[[], Any],
    local_rag_provider: Callable[[], Any],
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/chat")
    async def chat(message: str = Form(...), history: str = Form("[]")):
        try:
            history_list = json.loads(history)
            messages = history_list + [{"role": "user", "content": message}]
            return StreamingResponse(
                llm_provider().stream_generate(messages),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/chat/stream")
    async def chat_stream(messages: List[Dict[str, Any]]):
        try:
            return StreamingResponse(
                llm_provider().stream_generate(messages),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/chat/rag")
    async def chat_with_rag(
        message: str = Form(...),
        history: str = Form("[]"),
        use_rag: bool = Form(False),
        top_k: int = Form(3),
        mode: str = Form("quick"),
    ):
        try:
            history_list = json.loads(history)
            messages = history_list + [{"role": "user", "content": message}]

            if use_rag:
                local_rag = local_rag_provider()
                retrieved_docs = local_rag.search(message, top_k)

                if retrieved_docs:
                    context_parts = []
                    for idx, doc in enumerate(retrieved_docs, 1):
                        source = doc.get("filename", "未知来源")
                        content = doc.get("text", "")
                        context_parts.append(f"【文档 {idx}】来源: {source}\n{content}")

                    context = "\n\n".join(context_parts)
                    system_prompt = (
                        "你是一个智能助手。请基于以下参考信息回答用户问题。\n\n"
                        f"参考信息:\n{context}\n\n"
                        "请根据以上参考信息来回答用户的问题。"
                        "如果参考信息不足以回答，你可以基于自己的知识来回答。"
                    )
                    messages = [
                        {"role": "system", "content": system_prompt}
                    ] + messages

            return StreamingResponse(
                llm_provider().stream_generate(messages, mode=mode),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/analyze")
    async def analyze_image(
        message: str = Form(...),
        files: List[UploadFile] = File(None),
    ):
        try:
            content_list = []

            if files:
                for file in files:
                    contents = await file.read()
                    file_base64 = base64.b64encode(contents).decode("utf-8")

                    filename = (file.filename or "").lower()
                    if filename.endswith(".pdf"):
                        media_type = "application/pdf"
                        content_type = "file"
                    else:
                        ext = filename.split(".")[-1]
                        media_type = f"image/{ext}"
                        if media_type == "image/jpg":
                            media_type = "image/jpeg"
                        content_type = "image"

                    content_list.append({
                        "type": content_type,
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": file_base64,
                        },
                    })

            content_list.append({"type": "text", "text": message})
            messages = [{"role": "user", "content": content_list}]

            return StreamingResponse(
                multimodal_provider().stream_analyze(messages),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router