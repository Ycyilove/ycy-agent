"""provider 选择 + 落盘 stream_answer 的 prompt。"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def select_provider(
    *,
    attachments: List[Dict[str, Any]],
    message: str,
    messages: Optional[List[Dict[str, str]]],
    mode: str,
    multimodal_provider,
    llm_provider,
):
    """有全多模态附件 → multimodal；否则 → LLM。

    返回 (provider_iterable, is_multimodal: bool)。
    """
    from .multimodal import build_multimodal_message, is_multimodal

    if attachments and all(is_multimodal(item) for item in attachments):
        content = build_multimodal_message(message, attachments)
        return multimodal_provider().stream_analyze(content), True

    provider = llm_provider().stream_generate(messages, mode="deep")

    # 落盘 stream_answer 的输入
    try:
        from infra.debug_log.llm import dump_llm_call
        full_prompt = "\n\n".join(
            f"[{m.get('role', 'user')}]\n{m.get('content', '')}"
            for m in (messages or [])
        )
        dump_llm_call(
            kind="stream_answer",
            prompt=full_prompt,
            response=None,
            model=None,
            extra={"mode": mode, "message": message[:200]},
        )
    except Exception:
        logger.exception("dump stream_answer prompt failed")

    return provider, False