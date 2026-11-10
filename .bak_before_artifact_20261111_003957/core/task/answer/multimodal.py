"""多模态判定与消息构造。从 core/task/service.py 抽出，无依赖。"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, List


def attachment_label(attachment: Dict[str, Any]) -> str:
    return attachment.get("filename") or "未命名附件"


def is_multimodal(attachment: Dict[str, Any]) -> bool:
    filename = attachment_label(attachment)
    content_type = (attachment.get("content_type") or "").lower()
    suffix = Path(filename).suffix.lower()
    return content_type.startswith("image/") or suffix == ".pdf"


def build_multimodal_message(
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
    return content