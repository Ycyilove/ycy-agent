"""Thin wrapper: actual implementation in tools/tools_def/knowledge.py."""

from .tools_def.knowledge import *  # noqa: F401,F403
from .tools_def.knowledge import (  # noqa: F401
    set_rag_provider,
    knowledge_stats,
    list_knowledge_sources,
    list_knowledge_resources,
    get_knowledge_resource,
    search_knowledge,
)