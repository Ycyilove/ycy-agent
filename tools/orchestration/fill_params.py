"""AgentOrchestrator 的填参 Mixin：_fill_params。

从 core.py 按行号提取（B8-b-1），方法体零改动。
依赖的辅助方法：
    - self._read_tool_schema / self._format_discovery_schema_brief
      （仍在 core.py，MRO 找到）
    - self._format_single_tool_schema（在 decision.py::DecisionMixin，MRO 找到）
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from infra.debug_log.llm import dump_llm_call

from .state import OrchestratorState

logger = logging.getLogger(__name__)


class FillParamsMixin:
    """填参相关方法。self 是 AgentOrchestrator。"""
    async def _fill_params(
        self,
        goal: str,
        tool_name: str,
        tools: Dict[str, Any],
        state: Optional[OrchestratorState] = None,
        prior_messages: Optional[List[Dict[str, str]]] = None,
        raw_goal: Optional[str] = None,
        search_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """阶段 2：给单个工具填参数。

        路径参数不交给 LLM 猜——prompt 里给 workspace 真实路径，
        LLM 只需按 schema 填内容类字段。
        """
        meta = tools.get(tool_name)
        if meta is None:
            logger.warning(
                "[dsh][orchestrator] _fill_params: %s not in tools", tool_name
            )
            return {}

        _props, _required = self._read_tool_schema(meta)
        _is_mcp = tool_name.startswith("mcp__")
        _is_discovery = tool_name in self._MCP_DISCOVERY_TOOLS

        if (
            not _props and not _required
            and not _is_discovery and not _is_mcp
        ):
            logger.info(
                "[dsh][orchestrator] _fill_params: %s has no params, "
                "skip LLM call",
                tool_name,
            )
            return {}

        if (_is_discovery or _is_mcp) and not _props:
            _props = getattr(meta, "parameters", None) or {}
            if not _required and hasattr(meta, "required"):
                _required = list(getattr(meta, "required", []) or [])
            logger.info(
                "[dsh][orchestrator] _fill_params: %s schema empty, "
                "fallback to parameters: %r, required: %r",
                tool_name, list(_props.keys()), _required,
            )

        schema_text = self._format_single_tool_schema(meta)

        if _is_discovery:
            schema_text = self._format_discovery_schema_brief(meta)

        tool_desc = getattr(meta, "description", "") or ""

        history_text = ""
        if state is not None and state.history:
            compact_steps = []
            for h in state.history[-5:]:
                idx = h.find(" → ")
                if idx > 0:
                    compact_steps.append(h[:idx])
                else:
                    compact_steps.append(h[:200])
            history_text = (
                "## 已执行步骤（只列动作，产出见下方文件路径）\n"
                + "\n".join(compact_steps)
                + "\n\n"
            )

        NEEDS_FILE_CONTENT_TOOLS = {
            "edit_file", "write_file",
            "mcp__filesystem__edit_file", "mcp__filesystem__write_file",
        }
        file_content_block = ""
        if (
            tool_name in NEEDS_FILE_CONTENT_TOOLS
            and state is not None
            and state.last_read_content
        ):
            content = state.last_read_content
            path = state.last_read_path or "(未知)"
            if len(content) > 4000:
                total_lines = content.count("\n") + 1
                file_content_block = (
                    f"## 最近读取的文件\n"
                    f"路径：`{path}`（{total_lines} 行，{len(content)} 字符）\n"
                    f"**文件较大，未全部放入 prompt。两种修改方式**：\n"
                    f"1. **按行替换（推荐）**：先用 `read_file_range` 读目标行范围，"
                    f"确认后调用 `edit_file` 传 `start_line`/`end_line`/`new_text`。\n"
                    f"2. **逐字替换**：`old_text` 必须逐字匹配且唯一。\n"
                    f"前 2000 字符预览：\n```\n{content[:2000]}\n```\n\n"
                )
            else:
                file_content_block = (
                    f"## 最近读取的文件内容\n"
                    f"路径：`{path}`\n"
                    f"内容（**`old_text` 必须逐字匹配这里的内容**）：\n"
                    f"```\n{content}\n```\n"
                )

        conversation_block = ""
        if prior_messages and tool_name not in self._MCP_DISCOVERY_TOOLS:
            last_user = None
            for m in reversed(prior_messages):
                if m.get("role") == "user" and (m.get("content") or "").strip():
                    last_user = m
                    break
            if last_user:
                content = (last_user.get("content") or "").strip()
                if len(content) > 400:
                    content = content[:400] + "..."
                conversation_block = (
                    "## 上一轮用户消息（供填参参考）\n"
                    f"- {content}\n\n"
                )

        mem0_block = ""
        if (
            tool_name not in self._MCP_DISCOVERY_TOOLS
            and state is not None
            and getattr(state, "session_id", None)
        ):
            try:
                from core.memory.mem0_layer import recall_file_paths
                known_paths = recall_file_paths(state.session_id, limit=5)
                known_paths = [
                    p for p in known_paths
                    if isinstance(p, str)
                    and p.strip()
                    and p.strip() not in {"/", "\\"}
                    and not re.fullmatch(r"[A-Za-z]:[\\/]?", p.strip())
                ]
                if known_paths:
                    mem0_block = (
                        "## 已知文件路径（本会话历史）\n"
                        + "\n".join(f"- `{p}`" for p in known_paths)
                        + "\n\n"
                    )
                    logger.info(
                        "[dsh][orchestrator] mem0 recalled %d paths for session %s",
                        len(known_paths), state.session_id,
                    )
            except Exception:
                logger.exception("recall_file_paths 失败")

        workspace_block = ""
        if state is not None:
            ws_roots = getattr(state, "workspace_roots", None)
            if not ws_roots:
                ws_roots = [os.getcwd()]
            if ws_roots:
                workspace_block = (
                    "## 工作区真实路径（重要）\n"
                    + "\n".join(f"- `{p}`" for p in ws_roots)
                    + "\n\n**文件工具（read_text_file / write_file / edit_file / "
                    "list_files / search_content 等）的 path 参数必须用上面的绝对路径。"
                    "不要用进程 cwd 或相对路径。**\n\n"
                )

        coverage_block = ""
        if tool_name == "search_knowledge" and state is not None:
            target_raw = getattr(state, "_target_files_cache", None) or set()
            target = {_norm_filename(f) for f in target_raw if f}
            covered = {_norm_filename(f) for f in state.covered_documents if f}
            searched = state.searched_queries
            if target:
                missing = target - covered
                coverage_block = (
                    "## 多文档覆盖进度（重要）\n"
                    f"- 目标文档全集（{len(target)} 个）：{sorted(target)}\n"
                    f"- 已覆盖（{len(covered)} 个）：{sorted(covered)}\n"
                    f"- **待覆盖（{len(missing)} 个）**：{sorted(missing)}\n"
                )
                if missing:
                    coverage_block += (
                        f"- **本次请对 `{sorted(missing)[0]}` 调用 search_knowledge**，"
                        f"query 用该文档的主题或文件名\n"
                    )
                else:
                    coverage_block += "- **所有文档已覆盖，可以 done=true**\n"
                coverage_block += "\n"
            if searched:
                coverage_block += (
                    f"## 已搜索过的 query（不要重复）\n"
                    f"{sorted(searched)}\n\n"
                )

        discovery_block = ""
        if tool_name in self._MCP_DISCOVERY_TOOLS:
            goal_json = json.dumps(goal, ensure_ascii=False)
            failed = getattr(state, "failed_capabilities", set()) if state else set()
            failed_line = ""
            if failed:
                failed_line = (
                    f"- **已失败的关键词（不要再用）**："
                    f"{', '.join(sorted(failed))}\n"
                )
            discovery_block = (
                "## discovery 工具填参规则（强制，必须同时填两个字段）\n"
                f"- `user_query` = {goal_json}（原样，不翻译、不改写）\n"
                "- `capability` = 英文关键词，逗号分隔，2~4 个。\n"
                "  用 MCP 作者会在包名里用的通用词根。\n"
                f"{failed_line}"
                "- **两个字段都必须出现在 JSON 里**，缺一不可。\n"
                "- 输出示例：\n"
                f"  {{\"user_query\": {goal_json}, "
                "\"capability\": \"news,sports\", \"limit\": 100}}\n\n"
            )

        search_plan_block = ""
        _is_web_tool = (
            tool_name.startswith("mcp__tavily")
            or tool_name == "fetch_url"
            or tool_name.startswith("mcp__fetch")
        )
        if _is_web_tool and search_plan:
            _tiers = search_plan.get("source_tiers") or []
            _breadth = search_plan.get("breadth") or ""
            _queries = search_plan.get("queries") or []
            _lines = ["## 搜索覆盖计划（你上一步声明的，填参时请遵循）"]
            if _tiers:
                _lines.append(
                    f"- 目标来源类型：{_tiers}（official=官方/一手，"
                    f"news=新闻/媒体，forum=论坛/社区，blog=博客/个人）"
                )
            if _breadth:
                _lines.append(
                    f"- 覆盖深度：{_breadth}"
                    f"（narrow=1~2 次，medium=3~5 次，wide=6+ 次）"
                )
            if _queries:
                _lines.append(f"- 计划查询词：{_queries}")
                _lines.append(
                    "- **如果工具 schema 支持 `queries` 数组，直接填这个数组；"
                    "如果只支持单个 `query`，填第一个查询词**，"
                    "其余由编排层在后续步按计划继续搜。"
                )
            _lines.append(
                "- 如果 schema 有 domain 过滤参数，可把 official 类域名"
                "（如官方文档站）放进 include_domains。"
            )
            search_plan_block = "\n".join(_lines) + "\n\n"

        # ── 剥离 goal 开头的路径前缀，避免 LLM 把 goal 当 file_path ──
        def _strip_goal_path_prefix(text: str) -> str:
            if not text:
                return text
            m = re.match(
                r"^\s*(?:[A-Za-z]:[\\/]|/)[^\s]*?[\\/](?=[\u4e00-\u9fff]|\s|$)",
                text,
            )
            if m:
                stripped = text[m.end():].strip()
                if stripped:
                    return stripped
            return text

        task_text = _strip_goal_path_prefix(raw_goal or goal)
        prompt = (
            f"## 任务\n{task_text}\n\n"
            f"{conversation_block}"
            f"{mem0_block}"
            f"{workspace_block}"
            f"{coverage_block}"
            f"{history_text}"
            f"{file_content_block}"
            f"## 你要调用的工具\n"
            f"`{tool_name}`：{tool_desc}\n\n"
            f"{discovery_block}"
            f"{search_plan_block}"
            f"## 参数表\n{schema_text}\n\n"
            f"## 路径参数规则（强制）\n"
            f"1. **路径类参数（file_path / path / output_path）必须用"
            f"「工作区真实路径」里的绝对路径 + 一个合理文件名**。\n"
            f"2. **绝对不要从「任务」描述里抄路径**——任务描述里的路径"
            f"是用户上下文，不是输出路径。\n"
            f"3. **如果「工作区真实路径」为空，用 `output.<扩展名>` 作为文件名**，"
            f"不要用任务描述里的任何路径。\n"
            f"4. **绝对不要编造不存在的目录**（不要用 `D:/data`、`/tmp` 等占位）。\n\n"
            f"## 输出要求\n"
            f"- 只输出**一个** JSON 对象，代表**本次一次**工具调用。\n"
            f"- **禁止**输出多个 JSON 对象、JSON 数组。\n"
            f"- 不要任何解释，不要包裹在 markdown 代码块里。\n"
            f"- 确保 JSON 合法、字段名与**参数表完全一致**——"
            f"**只输出参数表里列出的字段**，不要凭经验添加其他字段。\n"
            + (
                f"如果参数表显示「该工具无参数」，输出 `{{}}`。\n"
                if not _props
                else (
                    f"示例（字段名仅供参考格式，**必须用参数表里的真实字段**）：\n"
                    f"```\n"
                    f"{{\"<参数名1>\": <值1>, \"<参数名2>\": <值2>}}\n"
                    f"```\n\n"
                )
            )
            + f"只输出 JSON："
        )

        import time as _time
        _t0 = _time.time()

        try:
            logger.info(
                "[dsh][orchestrator] _fill_params: calling LLM for %s, "
                "prompt length=%d",
                tool_name, len(prompt),
            )
            raw = await asyncio.wait_for(
                asyncio.to_thread(
                    self._llm_provider().generate,
                    [{"role": "user", "content": prompt}],
                    mode="quick",
                    timeout=110.0,
                ),
                timeout=120.0,
            )
            duration_ms = (_time.time() - _t0) * 1000
            logger.info(
                "[dsh][orchestrator] _fill_params raw response for %s: %r",
                tool_name, raw,
            )

            dump_llm_call(
                kind="fill_params",
                prompt=prompt,
                response=raw,
                model=getattr(self._llm_provider(), "model", None),
                duration_ms=duration_ms,
                tool_name=tool_name,
                extra={"goal": goal[:200]},
            )
        except ContextLengthExceeded as exc:
            duration_ms = (_time.time() - _t0) * 1000
            logger.warning(
                "[dsh][orchestrator] _fill_params context exceeded for %s: %s",
                tool_name, exc,
            )
            dump_llm_call(
                kind="fill_params", prompt=prompt, response=None,
                duration_ms=duration_ms, tool_name=tool_name,
                error=f"context_exceeded: {exc}",
            )
            fallback: Dict[str, Any] = {}
            if _required:
                for k in _required:
                    if k in ("query", "q", "keyword", "search_query"):
                        fallback[k] = goal
            return fallback
        except asyncio.TimeoutError:
            duration_ms = (_time.time() - _t0) * 1000
            logger.warning(
                "[dsh][orchestrator] _fill_params timed out (120s) for %s, "
                "fallback required params with goal",
                tool_name,
            )
            dump_llm_call(
                kind="fill_params",
                prompt=prompt,
                response=None,
                duration_ms=duration_ms,
                tool_name=tool_name,
                error="timeout(120s)",
            )
            fallback: Dict[str, Any] = {}
            if _required:
                for k in _required:
                    if k in ("query", "q", "keyword", "search_query"):
                        fallback[k] = goal
            return fallback
        except asyncio.CancelledError:
            logger.warning("[dsh][orchestrator] _fill_params cancelled")
            raise
        except Exception as exc:
            duration_ms = (_time.time() - _t0) * 1000
            logger.exception(
                "[dsh][orchestrator] _fill_params LLM failed for %s", tool_name
            )
            dump_llm_call(
                kind="fill_params",
                prompt=prompt,
                response=None,
                duration_ms=duration_ms,
                tool_name=tool_name,
                error=str(exc),
            )
            return {}

        data = self._parse_json(raw)
        if not isinstance(data, dict):
            logger.warning(
                "[dsh][orchestrator] _fill_params non-JSON for %s: %s",
                tool_name, (raw or "")[:200],
            )
            fallback: Dict[str, Any] = {}
            if _required:
                for k in _required:
                    if k in ("query", "q", "keyword", "search_query"):
                        fallback[k] = goal
            if fallback:
                logger.info(
                    "[dsh][orchestrator] _fill_params: %s parse failed, "
                    "fallback required params %r with goal",
                    tool_name, list(fallback.keys()),
                )
                return fallback
            return {}

        adapted = data

        allowed = set(_props.keys())
        if allowed:
            unknown = [k for k in adapted.keys() if k not in allowed]
            if unknown:
                logger.warning(
                    "[dsh][orchestrator] _fill_params: %s got unknown params "
                    "%r (allowed=%r), dropping",
                    tool_name, unknown, sorted(allowed),
                )
                adapted = {k: v for k, v in adapted.items() if k in allowed}
            missing = [k for k in _required if k not in adapted]
            if missing:
                logger.warning(
                    "[dsh][orchestrator] _fill_params: %s missing required %r",
                    tool_name, missing,
                )

        if _is_discovery:
            if not adapted.get("user_query"):
                adapted["user_query"] = goal
                logger.info(
                    "[dsh][orchestrator] _fill_params: %s user_query empty, "
                    "filled with goal", tool_name,
                )
            failed = getattr(state, "failed_capabilities", set()) if state else set()
            if failed:
                cap = adapted.get("capability", "")
                if cap:
                    kws = [
                        kw.strip() for kw in cap.split(",")
                        if kw.strip() and kw.strip().lower() not in failed
                    ]
                    if kws:
                        adapted["capability"] = ",".join(kws)
                        logger.info(
                            "[dsh][orchestrator] _fill_params: %s "
                            "filtered failed keywords, remaining: %r",
                            tool_name, kws,
                        )
                    else:
                        adapted.pop("capability", None)
                        logger.info(
                            "[dsh][orchestrator] _fill_params: %s "
                            "all keywords failed, cleared capability",
                            tool_name,
                        )

        if _required:
            missing = [k for k in _required if not adapted.get(k)]
            if missing:
                for k in missing:
                    if k in ("query", "q", "keyword", "search_query"):
                        adapted[k] = goal
                        logger.info(
                            "[dsh][orchestrator] _fill_params: %s "
                            "required %r empty, filled with goal",
                            tool_name, k,
                        )
                    elif k in ("path", "file_path"):
                        ws = (
                            getattr(state, "workspace_roots", None)
                            if state else None
                        )
                        _base_dir = ws[0] if ws else os.getcwd()
                        if "docx" in tool_name or "word" in tool_name:
                            _ext = ".docx"
                        elif "excel" in tool_name or "xlsx" in tool_name:
                            _ext = ".xlsx"
                        elif "pptx" in tool_name or "ppt" in tool_name:
                            _ext = ".pptx"
                        elif "pdf" in tool_name:
                            _ext = ".pdf"
                        else:
                            _ext = ".txt"
                        adapted[k] = os.path.join(_base_dir, f"output{_ext}")
                        logger.info(
                            "[dsh][orchestrator] _fill_params: %s "
                            "required %r empty, filled with default: %r "
                            "(workspace=%r)",
                            tool_name, k, adapted[k],
                            bool(ws),
                        )
                    elif k == "url":
                        import re as _re
                        m = _re.search(r'https?://[^\s"\']+', goal)
                        if m:
                            adapted[k] = m.group(0)

        for _pk in ("file_path", "path", "output_path"):
            _v = adapted.get(_pk)
            if isinstance(_v, str) and len(_v) > 200:
                logger.warning(
                    "[dsh][orchestrator] _fill_params: %s %s looks like "
                    "goal text (len=%d), dropping",
                    tool_name, _pk, len(_v),
                )
                adapted.pop(_pk, None)

        return adapted

    _VERIFY_TOOL_HINTS = (
        "run_test", "pytest", "test_runner", "jest", "vitest", "mocha",
        "run_python", "run_shell", "run_command", "run_script",
        "bash", "shell", "execute", "exec",
        "lint", "typecheck", "check",
    )

