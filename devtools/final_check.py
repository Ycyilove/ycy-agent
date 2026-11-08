"""端到端最终验收（静态版，零业务 import）。"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

FAILS: list[str] = []


def _run(name: str, fn) -> None:
    print(f"\n=== {name} ===")
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        FAILS.append(name)


def _read(rel: str) -> str:
    p = ROOT / rel
    if not p.is_file():
        raise AssertionError(f"文件不存在: {rel}")
    return p.read_text(encoding="utf-8")


def _top_level_imports(src: str) -> list[tuple[str, int, str]]:
    tree = ast.parse(src)
    out = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, node.lineno, a.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, node.lineno, node.module.split(".")[0]))
    return out


def check_deps() -> None:
    RULES = {
        # core 禁止依赖 tools/rag/mcp_client/backend，只能走 core.ports
        "core": {"tools", "rag", "mcp_client", "backend"},
        # tools 禁止依赖 backend
        "tools": {"backend"},
        # rag 禁止依赖 backend
        "rag": {"backend"},
        # mcp_client 禁止依赖 backend
        "mcp_client": {"backend"},
        # infra 只禁止依赖 backend（infra -> core/tools/rag/mcp_client 合法）
        "infra": {"backend"},
    }
    violations = []
    for pkg, forbidden in RULES.items():
        d = ROOT / pkg
        if not d.is_dir():
            continue
        for py in d.rglob("*.py"):
            try:
                imports = _top_level_imports(py.read_text(encoding="utf-8"))
            except SyntaxError as exc:
                violations.append(f"{py.relative_to(ROOT)}: 语法错误 {exc}")
                continue
            for mod, lineno, top in imports:
                if top in forbidden:
                    violations.append(f"{py.relative_to(ROOT)}:{lineno} {pkg} -> {mod}")
    if violations:
        raise AssertionError("依赖方向违规：\n" + "\n".join(violations))


def check_imports() -> None:
    local_pkgs = {"backend", "core", "tools", "rag", "mcp_client", "infra"}
    errors = []
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT)
        if rel.parts[0] in {".venv", "venv", "node_modules", "models", "data", "logs"}:
            continue
        try:
            imports = _top_level_imports(py.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            errors.append(f"{rel}: 语法错误 {exc}")
            continue
        for mod, lineno, top in imports:
            if top in local_pkgs and not (ROOT / top).is_dir():
                errors.append(f"{rel}:{lineno} 本地包不存在: {mod}")
    if errors:
        raise AssertionError("import 图问题：\n" + "\n".join(errors))


def check_b0() -> None:
    assert re.search(r"^from tools import get_registry\s*$", _read("tools/intent.py"), re.M), \
        "tools/intent.py 缺 from tools import get_registry"
    assert re.search(r"^import os\s*$", _read("tools/mcp_discovery.py"), re.M), \
        "mcp_discovery.py 缺 import os"
    assert re.search(r"^import time\s*$", _read("backend/routes/agent.py"), re.M), \
        "routes/agent.py 缺 import time"
    assert re.search(r"^import time\s*$", _read("backend/routes/trash.py"), re.M), \
        "routes/trash.py 缺 import time"

    src = _read("mcp_client/manager.py")
    n = src.count("infer_essential_when failed for")
    assert n == 1, f"manager.py 有 {n} 处 infer_essential_when failed（期望 1）"

    assert "enable_intent_router" not in _read("tests/test_orchestration.py"), \
        "test_orchestration.py 仍传 enable_intent_router"

    src = _read("backend/config.py")
    assert "all-MiniLM-L6-v2" in src, "config.py SHORTLIST_EMBEDDING_MODEL 默认值未改"
    assert "LANGFUSE_BASE_URL" in src, "config.py LANGFUSE_HOST 未兼容 LANGFUSE_BASE_URL"

    src = _read("backend/app.py")
    n = sum(1 for ln in src.splitlines() if ln.strip() == "mcp_worker = worker")
    assert n == 1, f"app.py 有 {n} 处 mcp_worker = worker（期望 1）"

    assert re.search(
        r"def __init__\(self,\s*ttl_seconds:\s*int\s*=\s*3600\)",
        _read("tools/security/approval.py"),
    ), "approval.py ttl_seconds 不是 3600"

    src = _read("tools/agent.py")
    assert "__approval_id__" in src, "agent.py 缺 __approval_id__"
    assert "get_approval_store().verify" in src, "agent.py 缺 get_approval_store().verify"

    assert "approved_approval_ids" in _read("core/task/service.py"), \
        "service.py 缺 approved_approval_ids 注入"

    for rel in [
        "tools/intent.py", "tools/mcp_discovery.py",
        "backend/routes/agent.py", "backend/routes/trash.py",
        "mcp_client/manager.py", "tests/test_orchestration.py",
        "backend/config.py", "backend/app.py",
        "tools/security/approval.py", "tools/agent.py",
        "core/task/service.py",
    ]:
        ast.parse(_read(rel))


# ── 6. B1：core/task/answer/ 拆分验收 ──

_B1_FILES = [
    "core/task/answer/__init__.py",
    "core/task/answer/multimodal.py",
    "core/task/answer/prompt_build.py",
    "core/task/answer/provider_call.py",
    "core/task/answer/splitter_bridge.py",
    "core/task/answer/asset_resolve.py",
    "core/task/answer/fallback_search.py",
    "core/task/answer/orchestrator.py",
]


def check_b1() -> None:
    """B1 验收：8 文件存在 + _stream_answer 精简 + 方法序列完整。"""
    # 1. 8 文件存在且语法正确
    for rel in _B1_FILES:
        src = _read(rel)
        ast.parse(src)

    # 2. _stream_answer ≤ 60 行
    src = _read("core/task/service.py")
    tree = ast.parse(src)
    target = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "AgentTaskService"
        ):
            for child in node.body:
                if (
                    isinstance(child, ast.AsyncFunctionDef)
                    and child.name == "_stream_answer"
                ):
                    target = child
                    break
            break
    assert target is not None, "找不到 AgentTaskService._stream_answer"
    n_lines = (target.end_lineno or target.lineno) - target.lineno + 1
    assert n_lines <= 60, (
        f"_stream_answer 有 {n_lines} 行（期望 ≤60）"
    )

    # 3. _stream_answer 内部必须有 from core.task.answer import answer_stream
    body_src = src.splitlines()[target.lineno - 1 : target.end_lineno]
    body_text = "\n".join(body_src)
    assert "from core.task.answer import answer_stream" in body_text, (
        "_stream_answer 未调用 answer_stream"
    )
    assert "async for evt in answer_stream(" in body_text, (
        "_stream_answer 未 async for answer_stream"
    )

    # 4. AgentTaskService 方法序列完整（31 个方法里 5 个关键方法都在）
    methods = [
        n.name for n in target.__class__ and []
    ] if False else [
        c.name for c in (
            next(
                n for n in tree.body
                if isinstance(n, ast.ClassDef)
                and n.name == "AgentTaskService"
            )
        ).body
        if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for m in [
        "_stream_answer", "_execute_tool",
        "_process_orchestrator_event", "stream_task", "stream_approval",
    ]:
        assert m in methods, f"AgentTaskService 缺方法: {m}"

_APPROVAL_SNIPPET = r"""
import sys
from pathlib import Path
ROOT = Path(r"%ROOT%")
sys.path.insert(0, str(ROOT))
from tools.security.approval import ApprovalStore
s = ApprovalStore(ttl_seconds=60)
rec = s.create("t1", "s1", "write_file", {"file_path": "a.txt"})
assert s.decide(rec.approval_id, "s1", "allow") is True
assert s.decide(rec.approval_id, "s1", "allow") is False
assert s.verify(rec.approval_id, "write_file", {"file_path": "a.txt"}) is True
assert s.verify(rec.approval_id, "write_file", {"file_path": "b.txt"}) is False
print("approval OK")
"""


def check_approval() -> None:
    snippet = _APPROVAL_SNIPPET.replace("%ROOT%", str(ROOT).replace("\\", "\\\\"))
    r = subprocess.run([PY, "-c", snippet], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise AssertionError("approval 单测失败：\n" + (r.stdout or "") + "\n" + (r.stderr or ""))


def check_pytest() -> None:
    r = subprocess.run([PY, "-m", "pytest", "tests/", "-q", "--no-header"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise AssertionError("pytest 失败：\n" + (r.stdout or "")[-3000:] + "\n" + (r.stderr or "")[-2000:])


CHECKS = {
    "deps": check_deps, "imports": check_imports, "b0": check_b0, "b1": check_b1,
    "approval": check_approval, "pytest": check_pytest,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("only", nargs="?",
                        choices=list(CHECKS.keys()) + ["all"], default="all")
    args = parser.parse_args()
    targets = CHECKS if args.only == "all" else {args.only: CHECKS[args.only]}
    for name, fn in targets.items():
        _run(name, fn)
    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print(f"ALL PASSED ({len(targets)} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())