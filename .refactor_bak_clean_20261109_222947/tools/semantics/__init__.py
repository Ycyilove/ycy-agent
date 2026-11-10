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
from .intent import (
    IntentSet,
    classify as classify_intent,
    INTENT_GENERATE,
    INTENT_MODIFY,
    INTENT_READ,
    INTENT_QUERY,
    INTENT_DB,
    INTENT_CODE,
    INTENT_TUTORIAL,
    INTENT_CHITCHAT,
    INTENT_UNKNOWN,
)
from .essential import (
    ESSENTIAL_ACTIONS,
    ESSENTIAL_FORMATS,
    essential_from_capability_tags,
    infer_essential_when,
)

__all__ = [
    "INTENT_TAG_DESCRIPTIONS",
    "get_intent_tags",
    "get_intent_tag_description",
    "tool_basename",
    "find_tools_by_intent",
    "classify_failure",
    "get_failure_recovery_hint",
    "IntentSet",
    "classify_intent",
    "INTENT_GENERATE",
    "INTENT_MODIFY",
    "INTENT_READ",
    "INTENT_QUERY",
    "INTENT_DB",
    "INTENT_CODE",
    "INTENT_TUTORIAL",
    "INTENT_CHITCHAT",
    "INTENT_UNKNOWN",
    "ESSENTIAL_ACTIONS",
    "ESSENTIAL_FORMATS",
    "essential_from_capability_tags",
    "infer_essential_when",
]