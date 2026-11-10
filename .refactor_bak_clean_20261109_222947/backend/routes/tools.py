"""工具管理路由（只读）。

⚠️ 安全说明：
    - /api/tools/execute 和 /api/tools/download 已删除：
      它们允许客户端传 confirmed=true 绕过审批，且不校验 danger_level
      与 workspace，是 RCE 后门。所有工具执行必须走
      /api/agent/tasks/stream → orchestrator → SecurityGateway。
    - 本文件只保留只读接口：列表、详情、意图分析。
"""

import traceback
from typing import Any, Callable

from fastapi import APIRouter, Form, HTTPException


def create_tools_router(agent_provider: Callable[[], Any]) -> APIRouter:
    router = APIRouter()

    @router.get("/api/tools")
    async def list_tools():
        try:
            agent = agent_provider()
            tools_desc = agent.registry.get_tool_descriptions()
            return {
                "status": "success",
                "tools": tools_desc,
                "count": len(tools_desc),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/api/tools/{tool_name}")
    async def get_tool_info(tool_name: str):
        try:
            agent = agent_provider()
            metadata = agent.registry.get_metadata(tool_name)
            if not metadata:
                raise HTTPException(status_code=404, detail=f"工具不存在: {tool_name}")
            return {
                "status": "success",
                "tool": {
                    "name": metadata.name,
                    "description": metadata.description,
                    "parameters": metadata.parameters,
                    "category": metadata.category,
                    "danger_level": metadata.danger_level,
                    "examples": metadata.examples,
                    "origin": metadata.origin,
                },
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/tools/analyze")
    async def analyze_intent(message: str = Form(...)):
        """只做意图分析，不执行工具。"""
        try:
            agent = agent_provider()
            intent = agent.analyze_intent(message)
            result = {
                "needs_tool": intent.needs_tool,
                "intent_type": intent.intent_type,
                "confidence": intent.confidence,
                "reasoning": intent.reasoning,
            }
            if intent.needs_tool and intent.suggested_tool:
                tool_call = agent.prepare_tool_call(intent)
                if tool_call is not None and hasattr(tool_call, "tool_name"):
                    result["tool"] = {
                        "name": tool_call.tool_name,
                        "parameters": tool_call.parameters,
                        "needs_confirmation": tool_call.user_confirmation_needed,
                    }
            return {"status": "success", "analysis": result}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    return router