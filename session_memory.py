"""Compatibility facade for the moved session persistence service."""

from backend.services.session_memory import SessionMemory, get_session_memory, init_session_memory

__all__ = ["SessionMemory", "get_session_memory", "init_session_memory"]
