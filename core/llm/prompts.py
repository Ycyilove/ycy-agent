"""Prompt construction for AgentTaskService.

把 prompt 相关的纯函数逻辑从 task_service 里抽出来，便于单测和维护。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional





def attachment_label(attachment: Dict[str, Any]) -> str:
    """附件显示名。"""
    return attachment.get("filename") or "未命名附件"


def attachment_context(attachments: List[Dict[str, Any]]) -> str:
    """附件上下文。B15 起委托 core.attachments。"""
    from core.attachments import attachment_context as _ac
    return _ac(attachments)


_INSTRUCTIONS = {
    "ask": (
        "回答用户问题。本次可能已调用只读工具（搜索/知识库/抓取），"
        "工具执行结果见下文；请基于真实结果回答，不要编造数据。"
        "不要声称执行了写/删/移动操作——ask 模式只读。"
        "如果确实没有工具结果，明确说明无法获取，不要编造。"
    ),
    "agent": (
        "直接完成用户要求。需要信息时先调用工具获取真实数据，"
        "再基于真实结果回答；结果要简洁并说明已完成的动作。"
    ),
}


def build_prompt(
    message: str,
    mode: str,
    attachments: List[Dict[str, Any]],
    rag_context: Optional[str] = None,
    tool_context: Optional[List[str]] = None,
    extra_suffix: str = "",
) -> str:
    """构造最终发给 LLM 的 prompt。"""
    instruction = _INSTRUCTIONS.get(mode)
    if instruction is None:
        raise ValueError(f"非法 mode={mode!r}，必须是 {sorted(_INSTRUCTIONS)}")

    prompt = f"工作模式：{mode}\n{instruction}\n\n用户请求：{message}"

    context = attachment_context(attachments)
    if context:
        prompt += f"\n\n附件内容：\n{context}"

    if rag_context:
        prompt += f"\n\n知识库参考信息：\n{rag_context}"
        prompt += (
            "\n\n引用规则："
            "如果需要引用知识库中的图片或附件，"
            "只能使用参考信息中的 [[asset:资源ID]] 标记，不要编造 URL。"
        )

    if tool_context:
        prompt += "\n\n工具执行结果：\n" + "\n\n".join(tool_context)
        prompt += (
            "\n\n请基于以上工具返回的真实数据回答用户问题。"
            "不要编造数据，不要声称没有权限，不要重复调用已经成功的工具。"
        )
    else:
        prompt += (
            "\n\n【重要】本次没有成功执行任何工具调用。"
            "如果需要执行操作，请明确说明你无法执行或请求用户提供更多信息，"
            "**不要编造执行结果**。"
        )

    if extra_suffix:
        prompt += extra_suffix

    prompt += (
        "\n\n输出格式要求："
        "先用「思考：」开头给出推理过程，再用「答案：」开头给出最终回答。"
    )

    return prompt
