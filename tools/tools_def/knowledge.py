"""
Knowledge base tools: generic primitives for the Agent to query RAG.

These are read-only "safe" tools. The model orchestrates them freely
to answer questions like "randomly pick an image from the knowledge base"
without us writing task-specific endpoints.

The RAG service instance is injected at app startup via set_rag_provider().
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .. import tool


_rag_provider = None


def set_rag_provider(provider) -> None:
    """Inject the LocalRAGService provider at app startup."""
    global _rag_provider
    _rag_provider = provider


def _get_rag():
    if _rag_provider is None:
        raise RuntimeError("RAG service provider is not configured")
    return _rag_provider()


def _resource_to_dict(resource: Any) -> Dict[str, Any]:
    """Convert ResourceRecord → plain dict for JSON serialization."""
    return {
        "id": resource.id,
        "filename": resource.filename,
        "mime_type": resource.mime_type,
        "kind": resource.kind,
        "page_no": getattr(resource, "page_no", None),
        "source_filename": getattr(resource, "source_filename", ""),
    }


@tool(
    name="knowledge_stats",
    description=(
        "查看知识库统计：文档数、分块数、资源数。"
        "**不确定知识库是否有内容时，先用这个工具检查。**"
    ),
    parameters={},
    examples=["knowledge_stats()"],
    category="knowledge",
    danger_level="safe",
)
def knowledge_stats() -> Dict[str, Any]:
    """Return RAG knowledge base statistics."""
    try:
        rag = _get_rag()
        stats = rag.get_stats()
        return {"success": True, **stats}
    except Exception as error:
        return {"success": False, "error": str(error)}


@tool(
    name="list_knowledge_sources",
    description=(
        "列出知识库中所有已索引的文档。"
        "**用于查看有哪些文档可以供 search_knowledge 检索。**"
    ),
    parameters={},
    examples=["list_knowledge_sources()"],
    category="knowledge",
    danger_level="safe",
)
def list_knowledge_sources() -> Dict[str, Any]:
    """List all indexed source documents."""
    try:
        rag = _get_rag()
        stats = rag.get_stats()
        files = stats.get("files", [])
        return {
            "success": True,
            "count": len(files),
            "sources": [
                {
                    "filename": f.get("filename"),
                    "chunk_count": f.get("chunk_count", 0),
                }
                for f in files
            ],
        }
    except Exception as error:
        return {"success": False, "error": str(error)}


@tool(
    name="list_knowledge_resources",
    description=(
        "List images and attachments extracted from knowledge base documents. "
        "Optionally filter by source filename or resource kind (image/chart/attachment). "
        "Use this to enumerate available assets before picking one."
    ),
    parameters={
        "source_filename": {"type": "str", "description": "Optional source document filename"},
        "kind": {"type": "str", "description": "Optional resource kind: image / chart / attachment"},
        "limit": {"type": "int", "description": "Max number of resources to return, default 20"},
    },
    required=[],
    examples=[
        "list_knowledge_resources()",
        "list_knowledge_resources(kind='image', limit=5)",
    ],
    category="knowledge",
    danger_level="safe",
)
def list_knowledge_resources(
    source_filename: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """List resources with optional filtering."""
    try:
        rag = _get_rag()
        repository = rag.resource_manager.repository

        if source_filename:
            records = repository.resources_for_source(source_filename)
        else:
            records = repository.list_all()

        if kind:
            records = [r for r in records if r.kind == kind]

        records = records[: max(1, int(limit))]

        refs = [rag.resource_manager.to_ref(r) for r in records]
        return {
            "success": True,
            "count": len(refs),
            "resources": refs,
        }
    except Exception as error:
        return {"success": False, "error": str(error)}


@tool(
    name="get_knowledge_resource",
    description=(
        "Get a single knowledge base resource by its id. "
        "Returns the URL and markdown reference."
    ),
    parameters={
        "resource_id": {"type": "str", "description": "Resource id returned by list_knowledge_resources"},
    },
    required=["resource_id"],
    examples=["get_knowledge_resource(resource_id='abc123')"],
    category="knowledge",
    danger_level="safe",
)
def get_knowledge_resource(resource_id: str) -> Dict[str, Any]:
    """Fetch a single resource by id."""
    try:
        rag = _get_rag()
        record = rag.resource_manager.repository.get(resource_id)
        if record is None:
            return {"success": False, "error": f"Resource not found: {resource_id}"}
        return {
            "success": True,
            "resource": rag.resource_manager.to_ref(record),
        }
    except Exception as error:
        return {"success": False, "error": str(error)}


@tool(
    name="search_knowledge",
    description=(
        "检索本地知识库（用户导入的 PDF / TXT / DOCX 文档）。"
        "**当用户询问文档内容、事实性问题，或任何可能已导入的知识时，优先使用这个工具**。"
        "不要用 search_content 代替——那只搜文件系统，搜不到知识库。"
        "返回相关文本片段和关联资源（图片/附件）。\n\n"
        "**mode 选择规则（重要）**：\n"
        "- `auto`（默认）：大多数情况用这个。\n"
        "- `flat`：问「文档讲了什么」「某段内容」这类内容综述问题时用，"
        "会返回段落级 chunk，覆盖多个文档。\n"
        "- `record`：**仅当**问「某人属于某团队」「某表格某行的值」这类结构化关联问题时用。"
        "record 只返回表格行，会漏掉段落内容。\n\n"
        "**不要对「文档讲了什么」这类问题用 record**——record 只会返回表格，"
        "且容易集中在一个文档，漏掉其他文档的正文。\n\n"
        "**大结果处理**：如果返回结果过大（> 4000 tokens），系统会把它卸载到磁盘，"
        "返回里会带 `result_id` 和 `handle`。此时可用 `read_tool_result` 工具"
        "按 filename / start_index 分页读取完整 chunk，不要因为摘要里没有完整内容就认为工具失败。"
    ),
    parameters={
        "query": {"type": "str", "description": "Natural language search query"},
        "top_k": {"type": "int", "description": "Number of chunks to return, default 15"},
        "mode": {
            "type": "str",
            "description": "flat / record / auto，默认 auto",
        },
    },
    required=["query"],
    examples=[
        "search_knowledge(query='summary of the design doc', top_k=15)",
        "search_knowledge(query='杨灿垣所在团队的奖项', mode='record')",
    ],
    category="knowledge",
    danger_level="safe",
)
def search_knowledge(query: str, top_k: int = 15, mode: str = "auto") -> Dict[str, Any]:
    try:
        rag = _get_rag()
        results = rag.search(query, top_k, mode=mode)
        resources = rag.resource_manager.collect_refs(results)

        chunks = []
        search_timing = None
        for item in results:
            chunk_text = item.get("text", "")
            chunk_meta = item.get("metadata", {}) or {}
            if search_timing is None and chunk_meta.get("timing_ms"):
                search_timing = chunk_meta["timing_ms"]

            chunk_resources = item.get("resources", []) or []
            enriched_resources = []
            for r in chunk_resources:
                if not isinstance(r, dict):
                    continue
                enriched_resources.append({
                    **r,
                    "chunk_preview": chunk_text[:120],
                    "chunk_filename": item.get("filename", ""),
                })

            # chunks[].resources 按 id 去重（同一资源可能因 enrich 兜底重复）
            seen_ids = set()
            deduped_chunk_resources = []
            for r in enriched_resources:
                rid = r.get("id")
                if rid and rid not in seen_ids:
                    seen_ids.add(rid)
                    deduped_chunk_resources.append(r)

            chunks.append({
                "text": chunk_text,
                "filename": item.get("filename", ""),
                "resources": deduped_chunk_resources,
            })

        paths: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        for item in results:
            for p in item.get("paths", []) or []:
                if p not in paths:
                    paths.append(p)
            for w in item.get("warnings", []) or []:
                if w not in warnings:
                    warnings.append(w)

        if not chunks and mode == "record":
            warnings.append({
                "type": "record_mode_empty",
                "message": (
                    "record 模式未返回任何结果；文档可能没有结构化表格/列表，"
                    "或 field_schema 为空。建议改用 mode='flat' 或 'auto'。"
                ),
            })

        # ── 新增：命中 chunk 但无资源 ──
        if chunks and not resources:
            warnings.append({
                "type": "chunks_without_resources",
                "message": (
                    "命中 chunk 但没有关联资源。可能是文档本身无内嵌图片，"
                    "或资源提取失败。若怀疑提取失败，可调 "
                    "list_knowledge_resources 确认该文档是否有资源。"
                ),
            })

        result = {
            "success": True,
            "query": query,
            "mode": mode,
            "chunks": chunks,
            "resources": resources,
            "paths": paths,
            "warnings": warnings,
        }
        if search_timing:
            result["timing_ms"] = search_timing
        return result
    except Exception as error:
        return {"success": False, "error": str(error)}