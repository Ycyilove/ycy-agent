"""工具管理路由。"""

import csv
import io
import json
import traceback
from typing import Any, Callable

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import Response

from tools.agent import ToolCallRequest


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
                if tool_call is None:
                    raise ValueError(
                        f"prepare_tool_call returned None for tool: {intent.suggested_tool}"
                    )
                if not hasattr(tool_call, "tool_name"):
                    raise ValueError(
                        f"tool_call is not a ToolCallRequest, got: {type(tool_call)} - {tool_call}"
                    )
                result["tool"] = {
                    "name": tool_call.tool_name,
                    "parameters": tool_call.parameters,
                    "needs_confirmation": tool_call.user_confirmation_needed,
                }
            return {"status": "success", "analysis": result}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/tools/execute")
    async def execute_tool(
        tool_name: str = Form(...),
        parameters: str = Form("{}"),
        confirmed: bool = Form(False),
    ):
        try:
            agent = agent_provider()
            metadata = agent.registry.get_metadata(tool_name)
            if not metadata:
                raise HTTPException(status_code=404, detail=f"工具不存在: {tool_name}")

            if metadata.danger_level in ["medium", "high"] and not confirmed:
                return {
                    "status": "confirmation_needed",
                    "message": agent.generate_confirmation_message(
                        ToolCallRequest(
                            tool_name=tool_name,
                            parameters=json.loads(parameters),
                            confidence=1.0,
                            reasoning="",
                            user_confirmation_needed=True,
                        )
                    ),
                    "danger_level": metadata.danger_level,
                }

            params = json.loads(parameters)

            # 导出类工具直接返回下载响应
            if tool_name == "export_to_csv":
                data = params.get("data", [])
                columns = params.get("columns")
                if not data:
                    raise HTTPException(status_code=400, detail="数据为空")
                if columns is None:
                    columns = list(data[0].keys()) if data else []

                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(data)
                csv_content = output.getvalue()

                filename = params.get("file_path", "export.csv")
                if not filename.endswith(".csv"):
                    filename += ".csv"

                return Response(
                    content=csv_content,
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"',
                        "Content-Type": "text/csv; charset=utf-8",
                    },
                )

            tool_call = ToolCallRequest(
                tool_name=tool_name,
                parameters=params,
                confidence=1.0,
                reasoning="",
                user_confirmation_needed=False,
            )
            result = await agent.execute_tool_async(tool_call)
            formatted_result = agent.format_tool_result(result)

            return {
                "status": "success",
                "result": formatted_result,
                "raw_result": (
                    result.result
                    if isinstance(result.result, dict)
                    else {"output": str(result.result)}
                ),
                "success": result.success,
                "execution_time": result.execution_time,
            }
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/api/tools/download")
    async def download_file(
        tool_name: str = Form(...),
        parameters: str = Form(...),
        confirmed: bool = Form(False),
    ):
        try:
            agent = agent_provider()
            metadata = agent.registry.get_metadata(tool_name)
            if not metadata:
                raise HTTPException(status_code=404, detail=f"工具不存在: {tool_name}")

            params = json.loads(parameters)

            if tool_name == "export_to_csv":
                data = params.get("data", [])
                columns = params.get("columns")
                if not data:
                    raise HTTPException(status_code=400, detail="数据为空")
                if columns is None:
                    columns = list(data[0].keys()) if data else []

                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(data)
                csv_content = output.getvalue()

                filename = params.get("file_path", "export.csv")
                if not filename.endswith(".csv"):
                    filename += ".csv"

                return Response(
                    content=csv_content,
                    media_type="text/csv",
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"',
                        "Content-Type": "text/csv; charset=utf-8",
                    },
                )

            if tool_name == "run_python_code":
                tool_call = ToolCallRequest(
                    tool_name=tool_name,
                    parameters=params,
                    confidence=1.0,
                    reasoning="",
                    user_confirmation_needed=False,
                )
                result = await agent.execute_tool_async(tool_call)
                output_text = (
                    str(result.result) if result.success else f"Error: {result.error}"
                )
                return Response(
                    content=output_text,
                    media_type="text/plain",
                    headers={
                        "Content-Disposition": "attachment; filename=output.txt",
                        "Content-Type": "text/plain; charset=utf-8",
                    },
                )

            tool_call = ToolCallRequest(
                tool_name=tool_name,
                parameters=params,
                confidence=1.0,
                reasoning="",
                user_confirmation_needed=False,
            )
            result = await agent.execute_tool_async(tool_call)
            result_text = (
                json.dumps(result.result, ensure_ascii=False, indent=2)
                if result.success
                else f"Error: {result.error}"
            )
            return Response(
                content=result_text,
                media_type="application/json",
                headers={
                    "Content-Disposition": f'attachment; filename={tool_name}_result.json',
                    "Content-Type": "application/json; charset=utf-8",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router