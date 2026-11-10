"""E2E: real goal -> stream_task -> assert SSE.

Two paths:
    1. fallback path (orchestrator_provider=None, prepare_tool_call returns None)
    2. orchestrator path (mock orchestrator.run yields done)
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import MagicMock

import pytest

from core.task.service import AgentTaskService, with_sse_done


def _parse_sse(events: List[str]) -> List[Dict[str, Any]]:
    out = []
    for e in events:
        if not e.startswith("data: "):
            continue
        payload = e[6:].strip()
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


def _make_agent(prepare_tool_call_returns: Any = None) -> MagicMock:
    agent = MagicMock()
    agent.list_all_tools = lambda **kw: {}
    agent.clear_available_tools = lambda: None
    agent.format_tool_result = lambda r: "OK"
    agent.analyze_intent = lambda msg, use_llm=True: MagicMock()
    agent.prepare_tool_call = lambda intent: prepare_tool_call_returns
    return agent


def _make_llm(answer_text: str = "test answer") -> MagicMock:
    llm = MagicMock()
    llm.model = "fake-model"

    def _gen(messages, **kw):
        return answer_text

    def _stream(messages, **kw):
        yield f'data: {json.dumps({"text": "Answer: " + answer_text}, ensure_ascii=False)}\n\n'
        yield "data: [DONE]\n\n"

    llm.generate = _gen
    llm.stream = _stream
    llm.stream_generate = _stream
    return llm


@pytest.mark.asyncio
async def test_e2e_fallback_no_tool(tmp_path, fake_session):
    """orchestrator_provider=None + prepare_tool_call=None -> _stream_answer."""
    llm = _make_llm("e2e fallback answer")
    agent = _make_agent(prepare_tool_call_returns=None)

    svc = AgentTaskService(
        agent_provider=lambda: agent,
        llm_provider=lambda: llm,
        multimodal_provider=lambda: llm,
        session_provider=lambda: fake_session,
        orchestrator_provider=None,
    )

    events = []
    async for evt in with_sse_done(svc.stream_task(
        task_id="e2e-fb-1",
        message="hello, introduce yourself briefly",
        history=[],
        mode="agent",
        workspace=[str(tmp_path)],
        session_id="e2e-session-fb",
        auto_discover_mcp=False,
        attachments=[],
    )):
        events.append(evt)

    parsed = _parse_sse(events)
    types = {e["type"] for e in parsed}

    assert "task" in types, f"missing task event: {types}"
    assert "step" in types, f"missing step event: {types}"
    assert "done" in types, f"missing done event: {types}"

    done_evt = next((e for e in parsed if e["type"] == "done"), None)
    assert done_evt is not None
    assert done_evt.get("task_status") == "done", (
        f"done task_status should be 'done', got {done_evt.get('task_status')!r}"
    )
    assert events[-1].strip() == "data: [DONE]"

    task = fake_session.get_task("e2e-fb-1")
    assert task is not None
    assert task.get("status") == "done", f"task status should be 'done', got {task.get('status')!r}"


class _FakeOrchestrator:
    """mock orchestrator: run() yields one done event."""

    def __init__(self) -> None:
        self.run_called = False

    async def run(
        self,
        goal: str,
        *,
        allowed_danger_levels=None,
        allow_mcp_discovery: bool = False,
        workspace_paths=None,
        prior_messages=None,
        session_id: str = "",
        task_id: str = "",
        resume_state=None,
        resume_from_step: int = 1,
    ) -> AsyncIterator[Dict[str, Any]]:
        self.run_called = True
        yield {"type": "done", "reason": "done"}


@pytest.mark.asyncio
async def test_e2e_orchestrator_path(tmp_path, fake_session):
    """orchestrator_provider returns mock orchestrator -> run yields done -> _stream_answer."""
    llm = _make_llm("e2e orch answer")
    agent = _make_agent(prepare_tool_call_returns=None)
    orch = _FakeOrchestrator()

    svc = AgentTaskService(
        agent_provider=lambda: agent,
        llm_provider=lambda: llm,
        multimodal_provider=lambda: llm,
        session_provider=lambda: fake_session,
        orchestrator_provider=lambda: orch,
    )

    events = []
    async for evt in with_sse_done(svc.stream_task(
        task_id="e2e-orch-1",
        message="list files in current workspace",
        history=[],
        mode="agent",
        workspace=[str(tmp_path)],
        session_id="e2e-session-orch",
        auto_discover_mcp=False,
        attachments=[],
    )):
        events.append(evt)

    assert orch.run_called, "orchestrator.run was not called"
    parsed = _parse_sse(events)
    types = {e["type"] for e in parsed}
    assert "task" in types
    assert "done" in types

    done_evt = next((e for e in parsed if e["type"] == "done"), None)
    assert done_evt is not None
    assert done_evt.get("task_status") == "done"
    assert events[-1].strip() == "data: [DONE]"