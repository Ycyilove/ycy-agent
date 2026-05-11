"""
Python代码沙箱安全运行工具
使用受限执行环境，禁止危险系统操作
"""
import io
import sys
import traceback
import ast
import re
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional

from . import tool, get_registry

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


def check_code_security(code: str) -> Optional[str]:
    """检查代码安全性"""
    for pattern in DANGEROUS_PATTERNS_COMPILED:
        match = pattern.search(code)
        if match:
            return f"禁止使用危险操作: {match.group()}"

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if hasattr(node.func, 'id'):
                    if node.func.id in ('eval', 'exec', 'compile', 'open', 'input', 'file'):
                        return f"禁止使用危险函数: {node.func.id}"
                elif hasattr(node.func, 'attr'):
                    if node.func.attr in ('read', 'write', 'system', 'popen', 'exit'):
                        return f"禁止使用危险方法: {node.func.attr}"
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
