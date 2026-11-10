"""附件解析。延迟 import rag，避免 core -> rag 模块级依赖。

B15 起 core/llm/prompts.py 改用本模块，不再直接 import rag。
依赖方向：core -> attachments ->（延迟）rag
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# 纯文本扩展名：直接 decode，不走 rag
TEXT_EXTENSIONS = frozenset({
    ".csv", ".json", ".md", ".txt", ".xml", ".yaml", ".yml",
})

# 二进制文档扩展名：走 rag.DocumentParserFactory
BINARY_EXTENSIONS = frozenset({".docx", ".pdf"})


def attachment_label(attachment: Dict[str, Any]) -> str:
    """附件显示名。"""
    return attachment.get("filename") or "未命名附件"


def parse_attachment(attachment: Dict[str, Any]) -> str:
    """解析单个附件为文本。

    - 纯文本扩展名：直接 decode
    - .docx / .pdf：延迟 import rag.DocumentParserFactory
    - 其他：返回空串
    """
    filename = attachment_label(attachment)
    content = attachment.get("content") or b""
    suffix = Path(filename).suffix.lower()

    if suffix in TEXT_EXTENSIONS:
        try:
            return content.decode("utf-8", errors="replace")
        except Exception:
            logger.exception("[attachments] decode failed: %s", filename)
            return ""

    if suffix in BINARY_EXTENSIONS:
        try:
            # 延迟 import：函数内 import，避免 core 模块级依赖 rag
            from rag import DocumentParserFactory
            return DocumentParserFactory.parse_file(content, filename)
        except Exception:
            logger.exception("[attachments] rag parse failed: %s", filename)
            return ""

    return ""


def attachment_context(attachments: List[Dict[str, Any]]) -> str:
    """把多个附件拼成 prompt 片段。"""
    parts: List[str] = []
    for attachment in attachments:
        filename = attachment_label(attachment)
        text = parse_attachment(attachment)
        if text.strip():
            parts.append(f"【附件 {filename}】\n{text[:12000]}")
    return "\n\n".join(parts)


__all__ = [
    "TEXT_EXTENSIONS",
    "BINARY_EXTENSIONS",
    "attachment_label",
    "parse_attachment",
    "attachment_context",
]