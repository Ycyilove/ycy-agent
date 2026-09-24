# ─────────────────────────────────────────────────────────────
# 沙箱白名单路径：允许在指定目录下用 Python 读写文件
#
# 为什么加白名单：
#   - 原来完全禁止 open()，导致用户问"用 Python 创建文件"时被拒
#   - Agent 需要"用代码操作数据文件"的合理场景
#   - 完全放开又太危险
#
# 策略：
#   - 白名单内的路径允许 open/write
#   - 白名单外的路径仍然禁止
#   - 从环境变量 SANDBOX_ALLOWED_PATHS 读取（冒号/分号分隔）
# ─────────────────────────────────────────────────────────────

import os


def _load_allowed_paths() -> List[str]:
    """从环境变量加载允许的沙箱路径。"""
    raw = os.getenv("SANDBOX_ALLOWED_PATHS", "")
    if not raw:
        # 默认允许项目下的 data 目录
        default = os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        )
        return [default]
    sep = ";" if ";" in raw else ":"
    return [os.path.abspath(p.strip()) for p in raw.split(sep) if p.strip()]


_ALLOWED_SANDBOX_PATHS: List[str] = _load_allowed_paths()


def _is_path_allowed_for_sandbox(path_str: str) -> bool:
    """检查路径是否在沙箱白名单内。"""
    if not path_str:
        return False
    try:
        abs_path = os.path.abspath(path_str)
    except Exception:
        return False
    return any(
        abs_path == root or abs_path.startswith(root + os.sep)
        for root in _ALLOWED_SANDBOX_PATHS
    )

import sys
import traceback
import ast
import re
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional

from .. import tool, get_registry
from backend.services.sandbox_backend import get_backend as _get_e2b_backend

DANGEROUS_PATTERNS = [
    r'import\s+os\b',
    r'import\s+sys\b',
    r'import\s+subprocess\b',
    r'import\s+shutil\b',
    r'import\s+pty\b',
    r'from\s+os\s+import',
    r'from\s+sys\s+import',
    r'from\s+subprocess\s+import',
    r'from\s+shutil\s+import',
    r'__import__\s*\(\s*["\']os',
    r'__import__\s*\(\s*["\']sys',
    r'__import__\s*\(\s*["\']subprocess',
    r'eval\s*\(',
    r'exec\s*\(',
    r'compile\s*\(',
    r'open\s*\([^)]*[\'"]?[rwa]',
    r'file\s*\(',
    r'input\s*\(',
    r'raw_input\s*\(',
    r'socket\b',
    r'urllib\b',
    r'requests\b',
    r'httpx\b',
    r'ftplib\b',
    r'telnetlib\b',
    r'smtplib\b',
    r'poplib\b',
    r'imaplib\b',
    r'pickle\b',
    r'marshal\b',
    r'yaml\.load\b',
    r'yaml\.unsafe_load\b',
    r'\.read\b',
    r'\.write\b',
    r'\.delete\b',
    r'rmtree\b',
    r'remove\s*\(',
    r'unlink\s*\(',
    r'mkdir\s*\(',
    r'makedirs\s*\(',
    r'chmod\s*\(',
    r'chown\s*\(',
    r'fork\b',
    r'exit\s*\(',
    r'quit\s*\(',
    r'sys\.exit',
    r'os\.system',
    r'os\.popen',
    r'subprocess\.call',
    r'subprocess\.run',
    r'subprocess\.Popen',
    r'shutil\.rmtree',
    r'shutil\.move',
    r'shutil\.copy',
]

DANGEROUS_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]


class SandboxTimeout(Exception):
    """沙箱超时异常"""
    pass


class SandboxSecurityError(Exception):
    """沙箱安全违规异常"""
    pass


# 只对"危险但非文件"的操作保留禁止
_DANGEROUS_NON_FILE_PATTERNS = [
    r'import\s+subprocess\b',
    r'from\s+subprocess\s+import',
    r'__import__\s*\(\s*["\']subprocess',
    r'os\.system',
    r'os\.popen',
    r'subprocess\.call',
    r'subprocess\.run',
    r'subprocess\.Popen',
    r'shutil\.rmtree',
    r'import\s+socket\b',
    r'socket\b',
    r'urllib\b',
    r'requests\b',
    r'httpx\b',
    r'ftplib\b',
    r'smtplib\b',
    r'pickle\b',
    r'marshal\b',
    r'yaml\.load\b',
    r'yaml\.unsafe_load\b',
    r'rmtree\b',
    r'unlink\s*\(',
    r'chmod\s*\(',
    r'chown\s*\(',
    r'fork\b',
    r'exit\s*\(',
    r'quit\s*\(',
    r'sys\.exit',
    r'eval\s*\(',
    r'exec\s*\(',
    r'compile\s*\(',
    r'input\s*\(',
]

_DANGEROUS_NON_FILE_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in _DANGEROUS_NON_FILE_PATTERNS
]

# 匹配 open(...) 调用，用于提取路径做白名单检查
_OPEN_CALL_PATTERN = re.compile(
    r'open\s*\(\s*(["\'])([^"\']+)\1',
    re.IGNORECASE,
)


def check_code_security(code: str) -> Optional[str]:
    """检查代码安全性（含沙箱路径白名单）。"""
    # 1. 检查非文件危险模式
    for pattern in _DANGEROUS_NON_FILE_COMPILED:
        match = pattern.search(code)
        if match:
            return f"禁止使用危险操作: {match.group()}"

    # 2. 检查文件访问是否在白名单内
    for match in _OPEN_CALL_PATTERN.finditer(code):
        path_str = match.group(2)
        if not _is_path_allowed_for_sandbox(path_str):
            return (
                f"沙箱外路径禁止访问: {path_str}\n"
                f"允许的路径: {_ALLOWED_SANDBOX_PATHS}"
            )

    # 3. AST 检查：禁止 eval/exec/compile 等
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'id'):
                    if node.func.id in ('eval', 'exec', 'compile', 'input', 'file'):
                        return f"禁止使用危险函数: {node.func.id}"
    except SyntaxError as e:
        return f"语法错误: {str(e)}"

    return None

class SafeGlobals:
    """安全全局命名空间"""
    ALLOWED_BUILTINS = {
        'print': print,
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'sorted': sorted,
        'reversed': reversed,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,
        'pow': pow,
        'divmod': divmod,
        'all': all,
        'any': any,
        'isinstance': isinstance,
        'issubclass': issubclass,
        'type': type,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,
        'slice': slice,
        'format': format,
        'hex': oct,
        'bin': bin,
        'ord': ord,
        'chr': chr,
        'bool': bool,
    }

    ALLOWED_MATH = {
        'pi': 3.141592653589793,
        'e': 2.718281828459045,
        'tau': 6.283185307179586,
        'inf': float('inf'),
    }

    @classmethod
    def get_globals(cls) -> Dict[str, Any]:
        """获取安全的全局命名空间"""
        return {
            '__builtins__': cls.ALLOWED_BUILTINS,
            **cls.ALLOWED_MATH,
        }


class ExecutionResult:
    """执行结果容器"""

    def __init__(self, success: bool, output: str = "", error: str = None, execution_time: float = 0.0):
        self.success = success
        self.output = output
        self.error = error
        self.execution_time = execution_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": f"{self.execution_time:.3f}s"
        }

    def __str__(self) -> str:
        if self.success:
            return f"[执行成功 | {self.execution_time:.3f}s]\n{self.output}"
        else:
            return f"[执行失败 | {self.execution_time:.3f}s]\n错误: {self.error}\n{self.output}"


@tool(
    name="run_python_code",
    description="在安全沙箱中执行Python代码，支持数学运算、数据处理、字符串操作等。禁止访问文件、网络、系统命令等危险操作。",
    parameters={
        "code": {"type": "str", "description": "要执行的Python代码"},
        "timeout": {"type": "int", "description": "执行超时时间（秒），默认10秒"}
    },
    examples=[
        "run_python_code(code='print(1+2+3)')",
        "run_python_code(code='result = [x**2 for x in range(10)]; print(sum(result))')"
    ],
    category="sandbox",
    danger_level="safe"
)
def run_python_code(code: str, timeout: int = 10) -> ExecutionResult:
    """在安全沙箱中执行Python代码"""
    import time
    start_time = time.time()

    # ── E2B 分支（失败自动回退本地） ──
    e2b = _get_e2b_backend()
    if e2b is not None:
        try:
            payload = e2b.execute(code, timeout=timeout)
            return ExecutionResult(
                success=payload["success"],
                output=payload["output"],
                error=payload["error"],
                execution_time=payload["execution_time"],
            )
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "[sandbox] E2B 执行失败，回退本地沙箱"
            )
            # 继续走本地实现

    security_error = check_code_security(code)
    if security_error:
        return ExecutionResult(
            success=False,
            error=security_error,
            execution_time=time.time() - start_time
        )

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, SafeGlobals.get_globals())

        execution_time = time.time() - start_time
        output = stdout_capture.getvalue()

        if stderr_capture.getvalue():
            output += "\n[stderr]: " + stderr_capture.getvalue()

        return ExecutionResult(
            success=True,
            output=output.strip() if output.strip() else "代码执行完成（无输出）",
            execution_time=execution_time
        )

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        return ExecutionResult(
            success=False,
            output=stdout_capture.getvalue(),
            error=error_msg,
            execution_time=execution_time
        )
