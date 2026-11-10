"""[[asset:xxx]] 解析：把 rag_resources 归一化后交给 ResourceManager。"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def resolve_assets(
    answer_text: str,
    rag_resources: Optional[List[Dict[str, Any]]],
    rag_provider,
) -> str:
    if not rag_resources:
        return answer_text
    try:
        rag = rag_provider()
        normalized = []
        for r in rag_resources:
            rid = r.get("id")
            url = r.get("url") or ""
            if not rid or not url:
                continue
            normalized.append({
                "id": rid,
                "url": url,
                "filename": r.get("filename") or "",
                "kind": r.get("kind") or "image",
                "markdown": r.get("markdown")
                    or f"![{r.get('filename', '')}]({url})",
            })
        if normalized:
            return rag.resource_manager.resolve_asset_tags(
                answer_text, normalized
            )
    except Exception as error:
        logger.warning("解析 asset 标签失败: %s", error)
    return answer_text