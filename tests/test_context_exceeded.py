"""验证 provider 层把 context 超限转成 ContextLengthExceeded。"""
from unittest.mock import MagicMock

import pytest

from core.llm.provider import ContextLengthExceeded, ModelScopeLLM


def _make_llm_with_side_effect(exc: Exception) -> ModelScopeLLM:
    llm = ModelScopeLLM()
    llm.use_local = False
    llm.llm = MagicMock()
    llm.llm.invoke.side_effect = exc
    return llm


def test_generate_raises_context_exceeded_maximum_context():
    llm = _make_llm_with_side_effect(
        Exception("This model's maximum context length is 8192 tokens")
    )
    with pytest.raises(ContextLengthExceeded):
        llm.generate([{"role": "user", "content": "x"}], mode="quick")


def test_generate_raises_context_exceeded_too_many_tokens():
    llm = _make_llm_with_side_effect(
        Exception("Error code: 400 - too many tokens in the prompt")
    )
    with pytest.raises(ContextLengthExceeded):
        llm.generate([{"role": "user", "content": "x"}], mode="quick")


def test_generate_passthrough_other_exception():
    llm = _make_llm_with_side_effect(ValueError("network error"))
    with pytest.raises(ValueError):
        llm.generate([{"role": "user", "content": "x"}], mode="quick")