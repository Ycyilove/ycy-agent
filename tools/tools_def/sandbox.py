"""沙箱执行：**只允许 E2B 后端**。

⚠️ 安全决策：
    本地 `exec()` 沙箱（黑名单正则 + SafeGlobals）可被属性链逃逸
    （`().__class__.__bases__[0].__subclasses__()`）绕过，等同 RCE。
    因此本模块**不再提供本地执行路径**：
    - SANDBOX_BACKEND != "e2b" → 工具不注册（见 tools/loader.py）
    - E2B 初始化失败 → 工具不注册
    - E2B 执行失败 → 直接返回失败，**不降级到本地**

    如果部署环境没有 E2B，请改用其他专用工具（read_csv / write_excel 等）
    完成数据处理，而不是 run_python_code。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .. import tool
from infra.sandbox import get_backend as _get_e2b_backend

logger = logging.getLogger(__name__)


class ExecutionResult:
    """执行结果容器（保留，供 task_service 等旧引用兼容）。"""

    def __init__(self, success: bool, output: str = "", error: str = None,
                 execution_time: float = 0.0):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time

    def __str__(self) -> str:
        if self.success:
            return f"[执行成功 | {self.execution_time:.3f}s]\n{self.output}"
        return (
            f"[执行失败 | {self.execution_time:.3f}s]\n"
            f"错误: {self.error}\n{self.output}"
        )


def is_backend_available() -> bool:
    """E2B 后端是否可用（供 loader 条件注册）。"""
    return _get_e2b_backend() is not None


@tool(
    name="run_python_code",
    description=(
        "在 E2B 云端沙箱中执行 Python 代码，支持数学运算、数据处理、字符串操作。"
        "**仅在 E2B 后端可用时注册**；不可用时本工具不会出现在工具目录里。"
        "禁止访问宿主文件系统与网络。"
    ),
    parameters={
        "code": {"type": "str", "description": "要执行的Python代码"},
        "timeout": {"type": "int", "description": "执行超时时间（秒），默认10秒"},
    },
    required=["code"],
    examples=[
        "run_python_code(code='print(1+2+3)')",
        "run_python_code(code='result = [x**2 for x in range(10)]; print(sum(result))')",
    ],
    category="sandbox",
    danger_level="safe",
)
def run_python_code(code: str, timeout: int = 10) -> ExecutionResult:
    """在 E2B 沙箱中执行 Python 代码（无本地回退）。"""
    import time
    start_time = time.time()

    e2b = _get_e2b_backend()
    if e2b is None:
        # 正常不会走到这里（loader 条件注册），保底：
        return ExecutionResult(
            success=False,
            error=(
                "E2B 沙箱后端不可用，run_python_code 拒绝执行。"
                "请改用 read_csv / write_excel / read_pdf_text 等专用工具，"
                "或在 .env 配置 SANDBOX_BACKEND=e2b + E2B_API_KEY 后重启。"
            ),
            execution_time=time.time() - start_time,
        )

    try:
        payload = e2b.execute(code, timeout=timeout)
        return ExecutionResult(
            success=payload["success"],
            output=payload["output"],
            error=payload["error"],
            execution_time=payload["execution_time"],
        )
    except Exception as exc:
        # ── 安全：E2B 失败不降级到本地（本地 exec 已删除） ──
        logger.exception("[sandbox] E2B 执行失败")
        return ExecutionResult(
            success=False,
            error=f"E2B 执行异常: {type(exc).__name__}: {exc}",
            execution_time=time.time() - start_time,
        )