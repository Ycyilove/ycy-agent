"""Semantics subpackage: tool intent tags + failure classification."""

from .tags import (
    INTENT_TAG_DESCRIPTIONS,
    get_intent_tags,
    get_intent_tag_description,
    tool_basename,
    find_tools_by_intent,
)
from .failure import (
    classify_failure,
    get_failure_recovery_hint,
)

__all__ = [
    "INTENT_TAG_DESCRIPTIONS",
    "get_intent_tags",
    "get_intent_tag_description",
    "tool_basename",
    "find_tools_by_intent",
    "classify_failure",
    "get_failure_recovery_hint",
]