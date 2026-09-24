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
    description="Get knowledge base statistics: number of documents, chunks, and resources.",
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
    description="List all documents in the knowledge base.",
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
            with repository._lock:
                records = list(repository._resources.values())

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
        "Semantic search across the knowledge base. "
        "Returns top-k text chunks with their linked resources. "
        "Use this when you need to find information or context to answer a question."
    ),
    parameters={
        "query": {"type": "str", "description": "Natural language search query"},
        "top_k": {"type": "int", "description": "Number of chunks to return, default 3"},
    },
    examples=["search_knowledge(query='summary of the design doc', top_k=5)"],
    category="knowledge",
    danger_level="safe",
)
def search_knowledge(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Semantic search with linked resources."""
    try:
        rag = _get_rag()
        results = rag.search(query, top_k)
        resources = rag.resource_manager.collect_refs(results)
        chunks = [
            {
                "text": item.get("text", ""),
                "filename": item.get("filename", ""),
                "resources": item.get("resources", []),
            }
            for item in results
        ]
        return {
            "success": True,
            "query": query,
            "chunks": chunks,
            "resources": resources,
        }
    except Exception as error:
        return {"success": False, "error": str(error)}