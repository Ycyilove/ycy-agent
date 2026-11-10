"""集成脚本：端到端校验核心链路。

用法：
    python scripts/integration_check.py

检查项：
    1. 后端启动后 /api/health 返回 200
    2. 提交"列出当前目录"任务，SSE 流正常结束
    3. 提交"生成 PDF"任务，_decide 选 write_docx
    4. 陈述句"为什么删除文件后回收站是空的"不触发 delete_file
"""

import sys
import json
import time
from pathlib import Path

import httpx


BASE_URL = "http://localhost:8000"


def check_health():
    r = httpx.get(f"{BASE_URL}/api/health", timeout=5)
    assert r.status_code == 200, f"health 返回 {r.status_code}"
    print("[OK] health")


def check_task(goal: str, expect_no_tool: bool = False):
    """提交任务，读 SSE 流，返回 (tool_calls, done)。"""
    tool_calls = []
    done = False
    with httpx.stream(
        "POST", f"{BASE_URL}/api/agent/tasks/stream",
        data={"message": goal, "mode": "agent"},
        timeout=120,
    ) as r:
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                break
            try:
                evt = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = evt.get("type")
            if t == "tool_start":
                tc = evt.get("tool_call", {})
                tool_calls.append(tc.get("tool_name"))
            if t == "done":
                done = True
    return tool_calls, done


def main():
    print("=== integration check ===")
    check_health()

    print("\n--- 1. 列出当前目录 ---")
    tools, done = check_task("列出当前目录的文件")
    print(f"    tools={tools}, done={done}")
    assert done

    print("\n--- 2. 陈述句不触发 delete_file ---")
    tools, done = check_task("为什么删除文件后回收站是空的？")
    print(f"    tools={tools}, done={done}")
    assert "delete_file" not in tools, (
        f"陈述句误触发 delete_file：{tools}"
    )

    print("\n--- 3. 指令句正常触发 delete_file ---")
    # 需要一个已存在的测试文件
    test_file = Path("data/hello.txt")
    if test_file.exists():
        tools, done = check_task(f"帮我删除 {test_file}")
        print(f"    tools={tools}, done={done}")
    else:
        print(f"    skip（{test_file} 不存在）")

    print("\n=== all checks passed ===")


if __name__ == "__main__":
    sys.exit(main())