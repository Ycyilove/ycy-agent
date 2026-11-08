"""工具编码的唯一入口。

所有策略（shortlist / intent_search / hybrid）用同一个 build_tool_text，
保证同一工具在不同索引里的向量一致。

编码文本格式：
    `{name} {description} {tag1} {tag2} ...`

tag 来源：
    - ToolMetadata.intent_tags（tags.py 的工具→用途标签）
    - ToolMetadata.capability_tags（MCP server 声明或推断）
    - MCPToolSpec.capability_tags（MCP 工具）

两者并集，越全召回越准。
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

# 统一 embedding 模型（与 rag 一致）
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


def _collect_tags(obj: Any) -> List[str]:
    """从 ToolMetadata / MCPToolSpec 提取所有 tag。"""
    tags: List[str] = []

    # capability_tags（MCP 工具）
    caps = getattr(obj, "capability_tags", ()) or ()
    for t in caps:
        s = str(t).strip()
        if s and s not in tags:
            tags.append(s)

    # intent_tags（内置工具，从 tags.py 取）
    name = getattr(obj, "name", "") or getattr(obj, "public_name", "")
    if name:
        try:
            from tools.semantics.tags import get_intent_tags
            for t in get_intent_tags(name):
                s = str(t).strip()
                if s and s not in tags:
                    tags.append(s)
        except Exception:
            logger.exception("[encoder] get_intent_tags failed for %r", name)

    # essential_when（已在 vocab 词表空间）
    ess = getattr(obj, "essential_when", ()) or ()
    for t in ess:
        s = str(t).strip()
        if s and s not in tags:
            tags.append(s)

    return tags


def build_tool_text(obj: Any) -> str:
    """把 ToolMetadata / MCPToolSpec 拼成统一编码文本。

    格式：`{name} {description} {tag1} {tag2} ...`
    """
    name = getattr(obj, "name", "") or getattr(obj, "public_name", "") or ""
    desc = getattr(obj, "description", "") or ""
    tags = _collect_tags(obj)
    parts = [name, desc] + tags
    return " ".join(p for p in parts if p).strip()


def encode_texts(texts: List[str]) -> List[List[float]]:
    """批量编码。归一化向量（cosine）。"""
    from infra.embedding import get_encoder

    encoder = get_encoder(EMBED_MODEL, local_files_only=True)
    vecs = encoder.encode(
        texts, normalize_embeddings=True, show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


def encode_query(query: str) -> List[float]:
    """编码单条 query。"""
    return encode_texts([query])[0]