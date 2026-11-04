"""回放 task-1793260822734 的决策序列，验证拆除特化后不再无限 read。

不依赖真实 LLM、不依赖真实工具。用 fake provider 按预设决策驱动 run()。

断言：
    1. 5 个目标文件都被读过
    2. 连续 read_text_file 不超过 5 次
    3. 所有读在第一次写之前完成
    4. 进入写阶段（edit_file / write_file）
    5. goal 含验证要求时，写后自动触发 run_tests

用法：
    python scripts/replay_task_1793260822734.py
    $env:LONG="1"; python scripts/replay_task_1793260822734.py   # 压预算，验证截断
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import MagicMock

# 允许从项目根直接跑
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 跳过 mem0 等待（脚本环境无 mem0，避免 30s 超时）
os.environ.setdefault("DSH_SKIP_MEM0_WAIT", "1")

from tools.agent import ToolCallResult
from tools.orchestration.core import AgentOrchestrator, OrchestratorStep


GOAL = (
    "请帮我修复以下问题，并确保 tests/ 下的测试全部通过："
    "1. calculator/core.py 的 divide 函数遇到除零会崩溃 "
    "2. calculator/validators.py 的 validate_positive 逻辑反了 "
    "3. calculator/formatters.py 的 format_result 精度不对 "
    "4. config.py 的 DATA_DIR 硬编码 "
    "5. main.py 调用 divide 时未导入 "
    "修复后请运行 tests/ 验证。"
)

WORKSPACE = ["project"]

# 预设决策序列：前 5 步读 5 个文件，之后应由 audit 逼出写操作
DECISIONS = [
    {"thought": "读 core.py", "done": False, "tool_name": "read_text_file"},
    {"thought": "读 validators.py", "done": False, "tool_name": "read_text_file"},
    {"thought": "读 formatters.py", "done": False, "tool_name": "read_text_file"},
    {"thought": "读 config.py", "done": False, "tool_name": "read_text_file"},
    {"thought": "读 main.py", "done": False, "tool_name": "read_text_file"},
    # ↓ 拆除生效后，第 6 步应转写
    {"thought": "改 core.py", "done": False, "tool_name": "edit_file"},
    {"thought": "改 validators.py", "done": False, "tool_name": "edit_file"},
    {"thought": "改 formatters.py", "done": False, "tool_name": "edit_file"},
    {"thought": "改 config.py", "done": False, "tool_name": "edit_file"},
    {"thought": "改 main.py", "done": False, "tool_name": "edit_file"},
    # run_tests 由 _collect_verification 自动触发，不在决策里
    {"thought": "完成", "done": True},
]

# 每个文件的"读结果"
_FILE_CONTENTS = {
    "core.py": 'def divide(a, b):\n    return a / b\n',
    "validators.py": 'def validate_positive(v):\n    if v > 0: raise ValueError()\n    return v\n',
    "formatters.py": 'def format_result(v):\n    return f"{v:.1f}"\n',
    "config.py": 'DATA_DIR = "D:/hardcoded/path"\n',
    "main.py": 'from calculator import add\nprint(divide(1, 2))\n',
}

# 读文件顺序（与 DECISIONS 前 5 步对应）
_READ_ORDER = ["core.py", "validators.py", "formatters.py", "config.py", "main.py"]

# 工具 schema（让 _fill_params 不走零参数早退）
_TOOL_SCHEMAS = {
    "read_text_file": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "文件路径"},
            "encoding": {"type": "string", "description": "编码，默认 utf-8"},
        },
        "required": ["file_path"],
    },
    "read_file_range": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        "required": ["file_path", "start_line", "end_line"],
    },
    "write_file": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "content": {"type": "string"},
            "encoding": {"type": "string"},
        },
        "required": ["file_path", "content"],
    },
    "edit_file": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["file_path", "old_text", "new_text"],
    },
    "run_tests": {
        "type": "object",
        "properties": {"project_dir": {"type": "string"}},
        "required": ["project_dir"],
    },
    "search_and_connect_mcp": {
        "type": "object",
        "properties": {
            "user_query": {"type": "string"},
            "capability": {"type": "string"},
        },
        "required": ["user_query"],
    },
}


class FakeLLM:
    """按预设序列返回决策；fill_params 按工具名返回参数。"""

    def __init__(self, decisions):
        self._it = iter(decisions)
        self._read_idx = 0
        self._edit_idx = 0
        self.calls = []
        self._stopped = False

    def generate(self, messages, mode="quick", timeout=None):
        prompt = messages[0]["content"] if messages else ""

        # decide 阶段
        if "多步工具编排助手" in prompt:
            try:
                d = next(self._it)
            except StopIteration:
                if self._stopped:
                    return json.dumps({"thought": "stop", "done": True}, ensure_ascii=False)
                self._stopped = True
                return json.dumps(
                    {"thought": "决策序列耗尽，收尾", "done": True},
                    ensure_ascii=False,
                )
            self.calls.append(("decide", d.get("tool_name") or "done"))
            print(f"DECIDE -> {d}", flush=True)
            return json.dumps(d, ensure_ascii=False)

        # fill_params 阶段：从 prompt 里抓工具名
        for tool in ("read_text_file", "edit_file", "write_file",
                     "run_tests", "read_file_range"):
            if f"`{tool}`" in prompt:
                self.calls.append(("fill", tool))
                print(f"FILL {tool}", flush=True)
                if tool == "read_text_file":
                    name = _READ_ORDER[min(self._read_idx, len(_READ_ORDER) - 1)]
                    self._read_idx += 1
                    return json.dumps({
                        "file_path": f"project/calculator/{name}",
                        "encoding": "utf-8",
                    }, ensure_ascii=False)
                if tool == "read_file_range":
                    return json.dumps({
                        "file_path": "project/calculator/core.py",
                        "start_line": 1, "end_line": 50,
                    }, ensure_ascii=False)
                if tool == "edit_file":
                    self._edit_idx += 1
                    target = _READ_ORDER[min(self._edit_idx - 1, len(_READ_ORDER) - 1)]
                    return json.dumps({
                        "file_path": f"project/calculator/{target}",
                        "old_text": "return a / b",
                        "new_text": "if b == 0: raise ZeroDivisionError(); return a / b",
                    }, ensure_ascii=False)
                if tool == "write_file":
                    return json.dumps({
                        "file_path": "project/calculator/core.py",
                        "content": "def divide(a, b):\n    if b == 0: raise ZeroDivisionError()\n    return a / b\n",
                    }, ensure_ascii=False)
                if tool == "run_tests":
                    return json.dumps({"project_dir": "project"}, ensure_ascii=False)

        print(f"FILL unknown: prompt head={prompt[:120]!r}", flush=True)
        return "{}"


def make_agent():
    """构造 fake agent：工具全部成功，记录调用序列。"""
    agent = MagicMock()
    exec_calls = []

    async def _exec(tc):
        exec_calls.append((tc.tool_name, dict(tc.parameters)))
        print(f"EXEC -> {tc.tool_name} {tc.parameters}", flush=True)
        if tc.tool_name == "run_tests":
            return ToolCallResult(
                tool_name="run_tests", success=True,
                result={"success": True, "passed": 5, "failed": 0},
            )
        if tc.tool_name == "read_text_file":
            fp = tc.parameters.get("file_path", "")
            base = os.path.basename(fp)
            return ToolCallResult(
                tool_name="read_text_file", success=True,
                result={
                    "content": _FILE_CONTENTS.get(base, "# file"),
                    "file_path": fp,
                    "size": 100,
                    "line_count": 10,
                    "success": True,
                },
            )
        if tc.tool_name in ("edit_file", "write_file"):
            return ToolCallResult(
                tool_name=tc.tool_name, success=True,
                result={
                    "success": True,
                    "file_path": tc.parameters.get("file_path", ""),
                },
            )
        return ToolCallResult(
            tool_name=tc.tool_name, success=True,
            result={"success": True},
        )

    agent.execute_tool_async = _exec
    agent.format_tool_result = lambda r: f"OK {r.tool_name}: {str(r.result)[:200]}"
    agent.register_dynamic_tool = lambda *a, **k: None
    agent.unregister_dynamic_tool = lambda *a, **k: None

    def _list_all_tools(allowed_danger_levels=None):
        def m(desc, schema):
            meta = MagicMock()
            meta.description = desc
            meta.input_schema = schema
            return meta
        return {
            name: m(name, _TOOL_SCHEMAS[name])
            for name in _TOOL_SCHEMAS
        }

    agent.list_all_tools = _list_all_tools
    return agent, exec_calls


def _install_bypasses():
    """绕过脚本环境不可用的外部依赖。"""
    # mem0
    try:
        from core.memory import mem0_layer
        async def _no_wait(*a, **k):
            return False
        mem0_layer.wait_until_ready = _no_wait
    except Exception:
        pass

    # skills
    try:
        from tools.skills import session_skills
        session_skills.build_skills_prompt_block = lambda *a, **k: ""
    except Exception:
        pass

    # reset_tried_servers
    try:
        from tools import mcp_discovery
        mcp_discovery.reset_tried_servers = lambda: None
    except Exception:
        pass

    print("[DBG] bypasses installed", flush=True)


async def main():
    _install_bypasses()

    # 预算覆盖（LONG=1 时压到 8000，验证 _assemble_decide_prompt 截断）
    budget_override = 8000 if os.getenv("LONG") == "1" else None

    agent, exec_calls = make_agent()
    fake_llm = FakeLLM(DECISIONS)

    orch = AgentOrchestrator(
        agent_provider=lambda: agent,
        llm_provider=lambda: fake_llm,   # 同一实例，迭代器不重置
        max_steps=12,
        enable_intent_router=False,      # 走 _decide，不走 fast path
    )
    if budget_override:
        orch._prompt_budget_chars = budget_override
        print(f"[LONG] budget={budget_override}", flush=True)

    events = []
    async for evt in orch.run(
        GOAL,
        allowed_danger_levels={"safe", "medium", "high"},
        allow_mcp_discovery=False,
        workspace_paths=WORKSPACE,
        session_id="replay-session",
        task_id="replay-task",
    ):
        events.append(evt)
        t = evt.get("type")
        if t == "tool_start":
            print(f"  [tool_start] {evt['tool_call'].tool_name}", flush=True)
        elif t == "thinking" and "结果评估" in str(evt.get("thought", "")):
            print(f"  [assess] {evt['thought'][:120]}", flush=True)
        elif t == "done":
            print(
                f"DONE: reason={evt.get('reason')} "
                f"detail={evt.get('detail', '')[:80]}",
                flush=True,
            )
            break
        elif t == "max_steps":
            print(f"MAX_STEPS at {evt.get('steps')}", flush=True)
            break
        elif t == "error":
            print(f"ERROR: {evt.get('message')}", flush=True)
            break

    # ── 断言 ──
    tool_seq = [name for name, _ in exec_calls]
    print("\n工具调用序列:", tool_seq, flush=True)

    ok = True

    # 1. 5 个目标文件都被读过
    read_paths = {
        os.path.basename(p.get("file_path", ""))
        for n, p in exec_calls
        if n == "read_text_file" and p.get("file_path")
    }
    expected = {"core.py", "validators.py", "formatters.py", "config.py", "main.py"}
    missing = expected - read_paths
    print(f"已读文件: {sorted(read_paths)}", flush=True)
    if missing:
        print(f"❌ 未读文件: {sorted(missing)}", flush=True)
        ok = False

    # 2. 连续 read_text_file 不超过 5
    max_run = 0
    cur = 0
    for name in tool_seq:
        if name == "read_text_file":
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    print(f"连续 read_text_file 最大次数: {max_run}", flush=True)
    if max_run > 5:
        print(f"❌ 连续 read 超过 5 次（{max_run}）", flush=True)
        ok = False

    # 3. 所有读在第一次写之前完成
    first_write_idx = next(
        (i for i, n in enumerate(tool_seq) if n in ("edit_file", "write_file")),
        -1,
    )
    reads_before_write = (
        sum(1 for n in tool_seq[:first_write_idx] if n == "read_text_file")
        if first_write_idx >= 0 else 0
    )
    print(f"第一次写前读了 {reads_before_write} 个文件", flush=True)
    if reads_before_write < 5:
        print(f"❌ 写之前只读了 {reads_before_write}/5 个文件", flush=True)
        ok = False

    # 4. 进入写阶段
    has_write = any(n in ("edit_file", "write_file") for n in tool_seq)
    print(f"进入写阶段: {has_write}", flush=True)
    if not has_write:
        print("❌ 未进入写阶段", flush=True)
        ok = False

    # 5. 写后触发验证
    has_verify = "run_tests" in tool_seq
    print(f"触发验证: {has_verify}", flush=True)
    if not has_verify:
        print("❌ 写后未触发验证", flush=True)
        ok = False

    print("\n✅ PASS" if ok else "\n❌ FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))