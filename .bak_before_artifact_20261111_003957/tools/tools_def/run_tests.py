"""在项目目录下运行 pytest，返回退出码与失败用例摘要。

用于 CODE_FIX 任务的完成质量校验：写完代码不等于修好了，
必须 tests 通过才算 DONE。
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, List

from tools import tool


_FAIL_LINE = re.compile(r"^(FAILED|ERROR)\s+(\S+)(?:\s+-\s+(.+))?$", re.M)
_PASS_SUMMARY = re.compile(r"(\d+)\s+passed", re.I)
_FAIL_SUMMARY = re.compile(r"(\d+)\s+failed", re.I)


@tool(
    name="run_tests",
    description=(
        "在项目目录下运行 pytest 测试。用于代码修复后验证是否真的修好。"
        "返回退出码、通过/失败数量、失败用例列表。"
    ),
    parameters={
        "project_dir": {
            "type": "string",
            "description": "项目根目录（绝对路径），如 D:/proj/foo",
        },
        "test_path": {
            "type": "string",
            "description": "测试路径，默认 tests/",
        },
        "timeout": {
            "type": "integer",
            "description": "超时秒数，默认 120",
        },
    },
    required=["project_dir"],
    category="code",
    danger_level="medium",
)
def run_tests(
    project_dir: str,
    test_path: str = "tests/",
    timeout: int = 120,
) -> Dict[str, Any]:
    """在 project_dir 下运行 pytest test_path。"""
    import sys
    cmd = [sys.executable, "-m", "pytest", test_path, "-q", "--no-header", "--tb=short"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False, "exit_code": -1, "passed": 0, "failed": 0,
            "failures": [], "stdout_tail": "",
            "message": f"pytest 超时（{timeout}s）",
        }
    except Exception as exc:
        return {
            "success": False, "exit_code": -1, "passed": 0, "failed": 0,
            "failures": [], "stdout_tail": "",
            "message": f"pytest 启动失败：{type(exc).__name__}: {exc}",
        }

    stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
    exit_code = proc.returncode

    passed = 0
    failed = 0
    m = _PASS_SUMMARY.search(stdout)
    if m:
        passed = int(m.group(1))
    m = _FAIL_SUMMARY.search(stdout)
    if m:
        failed = int(m.group(1))

    failures: List[Dict[str, str]] = []
    for fm in _FAIL_LINE.finditer(stdout):
        failures.append({
            "test": fm.group(2),
            "message": (fm.group(3) or "").strip(),
        })

    success = exit_code == 0
    return {
        "success": success,
        "exit_code": exit_code,
        "passed": passed,
        "failed": failed,
        "failures": failures[:20],
        "stdout_tail": stdout[-3000:],
        "message": (
            f"pytest {'通过' if success else '失败'}："
            f"{passed} passed, {failed} failed (exit={exit_code})"
        ),
    }