"""工具结果格式化（从 ToolAgent 抽出）。

从 ToolAgent.format_tool_result 拆出的纯函数；
唯一外部依赖是 available_tools 参数，不依赖实例状态。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .types import ToolCallResult
from .semantics import classify_failure, get_failure_recovery_hint
from .guidance import extract_guidance

logger = logging.getLogger(__name__)


def format_tool_result(
    result: ToolCallResult,
    *,
    available_tools: Optional[List[str]] = None,
) -> str:
    """格式化工具结果。

    ⚠️ 安全：MCP 返回不提取 guidance（防 prompt injection）。
    """
    guidance: Dict[str, Any] = {}
    # 只对内置工具提取 guidance
    if (
        isinstance(result.result, dict)
        and not result.tool_name.startswith("mcp__")
        and result.tool_name not in ("search_and_connect_mcp",)
    ):
        try:
            guidance = extract_guidance(result.result)
        except Exception:
            logger.exception("extract_guidance failed")

    next_tool = guidance.get("next_tool")
    # next_tool 必须在 available_tools 里
    if next_tool and available_tools and next_tool not in available_tools:
        logger.warning(
            "[format] guidance next_tool %r not in available, dropped", next_tool
        )
        next_tool = None
        guidance["guidance_type"] = "none"

    next_args = guidance.get("next_args") or {}
    next_message = guidance.get("message") or ""
    guest_tools = guidance.get("guest_tools") or []
    guidance_type = guidance.get("guidance_type", "none")

    # 补 mcp__server__ 前缀
    if next_tool and not next_tool.startswith("mcp__") and "__" not in next_tool:
        current = result.tool_name
        if current.startswith("mcp__"):
            parts = current.split("__", 2)
            if len(parts) == 3:
                next_tool = f"mcp__{parts[1]}__{next_tool}"

    directive_block = ""
    if next_tool and guidance_type == "mandatory":
        directive_block += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        directive_block += (
            "⛔ 任务未完成。必须先完成前置步骤：\n"
            if result.success
            else "❌ 工具调用被 MCP 服务器拒绝。\n"
        )
        directive_block += f"   1. 调用 {next_tool}\n"
        if next_args:
            directive_block += f"      参数: {json.dumps(next_args, ensure_ascii=False)}\n"
        if next_message:
            directive_block += f"      原因: {next_message[:300]}\n"
        directive_block += "   **在完成这一步之前，禁止调用其他工具。**\n"
        directive_block += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    elif next_tool and guidance_type == "optional":
        directive_block += "┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈\n"
        directive_block += "💡 MCP 服务器附带了一个可选建议（与当前目标可能无关）：\n"
        directive_block += f"   - 可尝试 {next_tool}"
        if next_args:
            directive_block += f"，参数 {json.dumps(next_args, ensure_ascii=False)}"
        directive_block += "\n"
        if next_message:
            directive_block += f"   - 服务器说明: {next_message[:300]}\n"
        directive_block += (
            "   **这是可选建议。如果它与用户目标无关，请忽略，"
            "直接采取其他方式达成目标。**\n"
        )
        directive_block += "┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈\n\n"

    # 成功分支
    if result.success:
        output = directive_block
        output += f"✅ 工具 [{result.tool_name}] 执行成功\n"
        output += f"⏱️ 耗时: {result.execution_time:.3f}s\n\n"

        if isinstance(result.result, dict):
            # ── 1. 连接类返回（discovery 工具特有） ──
            if "tools" in result.result and isinstance(result.result["tools"], list):
                tools_list = result.result["tools"]
                if tools_list:
                    output += "🔧 已连接 MCP 服务器。**下一步必须使用以下确切工具名**：\n"
                    for t in tools_list:
                        output += f"  - {t}\n"
                    output += "\n"

            # ── 2. structuredContent（MCP 2024-11 规范，优先级最高） ──
            structured = result.result.get("structuredContent")
            if structured is not None:
                try:
                    output += "📊 结构化数据:\n"
                    output += json.dumps(
                        structured, ensure_ascii=False, indent=2
                    )[:3000]
                    output += "\n"
                except Exception:
                    logger.exception("序列化 structuredContent 失败")

            # ── 3. content 数组：MCP 标准返回 ──
            content_blocks = result.result.get("content")
            if isinstance(content_blocks, list) and content_blocks:
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")

                    if btype == "text":
                        text = block.get("text") or ""
                        if not text:
                            continue
                        # 剥 HTML 注释
                        import re
                        text_clean = re.sub(
                            r"<!--.*?-->", "", text, flags=re.DOTALL
                        ).strip()
                        if not text_clean:
                            text_clean = text.strip()

                        # 尝试解析为 JSON
                        parsed = None
                        if text_clean.startswith(("{", "[")):
                            try:
                                parsed = json.loads(text_clean)
                            except (json.JSONDecodeError, ValueError):
                                parsed = None

                        if parsed is not None:
                            output += "📊 MCP 数据:\n"
                            try:
                                output += json.dumps(
                                    parsed, ensure_ascii=False, indent=2
                                )[:3000]
                            except Exception:
                                output += str(parsed)[:3000]
                            output += "\n"
                        else:
                            # 非 JSON（纯文本、Markdown）
                            output += f"📄 MCP 返回:\n{text_clean[:2000]}\n"

                    elif btype == "image":
                        output += "🖼️ [MCP 返回图片，base64 已省略]\n"

                    elif btype == "resource":
                        res = block.get("resource") or {}
                        uri = res.get("uri") or ""
                        output += f"📎 [资源: {uri}]\n"

            # ── 4. 无 content 数组时，退到通用字段 ──
            # 官方 MCP 只用 content / structuredContent；
            # message / output / data 是非标 server 的旧字段，
            # 这里不再逐个特判，统一走 leftover 兜底。
            if not isinstance(content_blocks, list) or not content_blocks:
                handled_keys = {
                    "content", "structuredContent",
                    "tools", "isError", "resultType", "success", "_meta",
                }
                leftover = {
                    k: v for k, v in result.result.items()
                    if k not in handled_keys
                }
                if leftover:
                    try:
                        output += "📦 其他数据:\n"
                        # chunks / resources 是知识库检索的核心内容，
                        # 放宽到 12000，避免人名/实体被截断
                        limit = 12000 if (
                            "chunks" in leftover or "resources" in leftover
                        ) else 4000
                        output += json.dumps(
                            leftover, ensure_ascii=False, indent=2
                        )[:limit]
                        output += "\n"
                    except Exception:
                        pass

            if guest_tools:
                output += f"\n✅ 无需认证的工具：{', '.join(guest_tools)}\n"
        else:
            output += f"📤 结果: {result.result}\n"

        return output

    # ── 失败分支：原文在前，hint 只做补充 ──
    error_message = result.error or "(未知错误)"
    failure_type = classify_failure(error_message)

    available: List[str] = list(available_tools or [])

    recovery_hint = get_failure_recovery_hint(
        failure_type=failure_type,
        error_message=error_message,
        failed_tool_name=result.tool_name,
        available_tools=available,
    )

    # 1. 内置工具 guidance（MCP 返回不提取）
    output = directive_block

    # 2. 失败原文 —— 核心信息放这里，不遮不掩
    output += f"❌ 工具 [{result.tool_name}] 执行失败\n"
    output += f"💥 错误: {error_message}\n"

    # 3. 详细返回（MCP content 原文）
    if isinstance(result.result, dict):
        content_blocks = result.result.get("content") or []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text") or ""
                if text and text != error_message and len(text) < 1500:
                    output += f"\n📄 详细返回:\n{text}\n"
                    break
        structured = result.result.get("structuredContent")
        if isinstance(structured, dict):
            try:
                output += (
                    f"\n📊 结构化数据:\n"
                    f"{json.dumps(structured, ensure_ascii=False, indent=2)[:2000]}\n"
                )
            except Exception:
                pass

    # 4. hint 放最后，只作为"额外信息"，不抢占注意力
    if recovery_hint:
        output += f"\n{recovery_hint}"

    return output
