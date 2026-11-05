# diagnose_tavily.py
"""一次性诊断：tavily 为什么没进 decide prompt 的工具目录。

用法（项目根目录）：
    python diagnose_tavily.py

不需要改任何项目代码；只读 registry / manager / 意图索引。
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import asyncio
import json

GOAL = "我想知道deepseek曾经出现过的所有服务器繁忙的宕机事件"


def _line(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


async def main() -> int:
    # ── 0. 导入项目模块（触发 backend.config 加载 .env） ──
    _line("0. 导入项目模块")
    try:
        import backend.config  # noqa: F401
        from tools import get_registry
        from tools.mcp_discovery import _get_manager
        print("  导入成功")
    except Exception as exc:
        print(f"  导入失败: {type(exc).__name__}: {exc}")
        return 1

    # ── 1. MCP manager：自建实例，不依赖 _get_manager 的全局 context ──
    _line("1. MCP manager：配置 / 连接状态")
    from mcp_client.config import load_mcp_config
    from mcp_client.manager import MCPClientManager

    configs = load_mcp_config()
    if not configs:
        _fail("load_mcp_config() 返回空 —— mcp_servers.json 不存在或解析失败")
        return 1
    print(f"  load_mcp_config() 读到 {len(configs)} 个 server: {[c.name for c in configs]}")
    manager = MCPClientManager(configs)
    print(f"  manager.configs: {sorted(manager.configs.keys())}")
    configs = getattr(manager, "configs", {}) or {}
    connections = getattr(manager, "connections", {}) or {}
    statuses = getattr(manager, "statuses", {}) or {}

    print(f"  configs: {sorted(configs.keys())}")
    print(f"  connections: {sorted(connections.keys())}")

    tavily_configured = "tavily" in manager.configs
    if tavily_configured:
        _ok("tavily 在 configs 里")
    else:
        _fail("tavily 不在 configs 里 —— mcp_servers.json 没配")

    # 手动 connect + discover（只诊断，不影响你正在跑的服务）
    try:
        from tools.agent import ToolAgent
        agent = ToolAgent()
        agent.set_mcp_client_manager(manager)
    except Exception as exc:
        _fail(f"ToolAgent 构造失败: {type(exc).__name__}: {exc}")
        agent = None

    if agent is not None and tavily_configured:
        print("\n  手动 connect + discover tavily ...")
        try:
            await asyncio.wait_for(manager.connect("tavily"), timeout=40)
            _ok("connect 成功")
        except Exception as exc:
            _fail(f"connect 失败: {type(exc).__name__}: {exc}")
        try:
            await asyncio.wait_for(
                manager.discover(
                    "tavily",
                    agent.register_dynamic_tool,
                    agent.unregister_dynamic_tool,
                ),
                timeout=40,
            )
            _ok("discover 成功")
        except Exception as exc:
            _fail(f"discover 失败: {type(exc).__name__}: {exc}")

    # 打印 statuses（自建 manager 的）
    for name, st in manager.statuses.items():
        if name == "tavily":
            print(f"  tavily status = {st}")

    # 如果没连上，尝试手动 connect + discover（只诊断，不改全局状态）
    if tavily_configured and not tavily_connected:
        print("\n  尝试手动 connect tavily ...")
        agent = None
        try:
            from tools.agent import ToolAgent
            agent = ToolAgent()
        except Exception as exc:
            print(f"  ToolAgent 构造失败: {exc}")
        if agent is not None:
            try:
                await asyncio.wait_for(manager.connect("tavily"), timeout=40)
                _ok("手动 connect 成功")
                await asyncio.wait_for(
                    manager.discover(
                        "tavily",
                        agent.register_dynamic_tool,
                        agent.unregister_dynamic_tool,
                    ),
                    timeout=40,
                )
                _ok("手动 discover 成功")
            except Exception as exc:
                _fail(f"手动 connect/discover 失败: {type(exc).__name__}: {exc}")

    # ── 2. registry 里有没有 tavily 工具，danger_level 是什么 ──
    _line("2. registry：tavily 工具是否注册，danger_level / origin")
    registry = get_registry()
    all_meta = registry.get_all_metadata()
    tavily_tools = {n: m for n, m in all_meta.items() if n.startswith("mcp__tavily")}
    if not tavily_tools:
        _fail("registry 里没有 mcp__tavily__* —— discover 没注册，或注册后被 unregister")
    else:
        for n, m in tavily_tools.items():
            print(
                f"  - {n}: origin={getattr(m, 'origin', None)!r}, "
                f"danger_level={getattr(m, 'danger_level', None)!r}, "
                f"idempotent={getattr(m, 'idempotent', None)!r}"
            )

    # ── 3. ask 模式 allowed_danger_levels={'safe'} 是否会过滤 tavily ──
    _line("3. allowed_danger_levels={'safe'} 过滤后 tavily 是否还在")
    try:
        from tools.agent import ToolAgent
        agent = ToolAgent()
        tools_safe = agent.list_all_tools(allowed_danger_levels={"safe"})
        tools_all = agent.list_all_tools(allowed_danger_levels=None)
        print(f"  list_all_tools(None)    count={len(tools_all)}")
        print(f"  list_all_tools({{'safe'}}) count={len(tools_safe)}")
        tavily_all = [n for n in tools_all if n.startswith("mcp__tavily")]
        tavily_safe = [n for n in tools_safe if n.startswith("mcp__tavily")]
        print(f"  tavily in all:  {tavily_all}")
        print(f"  tavily in safe: {tavily_safe}")
        if tavily_all and not tavily_safe:
            _fail(
                "tavily 注册了但 danger_level 不是 safe —— "
                "ask 模式（allowed={'safe'}）会把它过滤掉。"
                "查 manager.discover 的 readOnlyHint 是否缺失。"
            )
        elif tavily_safe:
            _ok("tavily 在 safe 工具集里 —— danger_level 过滤不是原因")
        elif not tavily_all:
            _fail("tavily 连 registry 都没进 —— 回到第 1/2 步")
    except Exception as exc:
        _fail(f"list_all_tools 检查失败: {type(exc).__name__}: {exc}")

    # ── 4. _select_shortlist 实际保留了什么 ──
    _line("4. _select_shortlist：MCP 工具保留情况")
    try:
        from tools.orchestration.core import AgentOrchestrator
        from tools.orchestration.state import OrchestratorState
        from tools.agent import ToolAgent

        agent = ToolAgent()
        orch = AgentOrchestrator(
            agent_provider=lambda: agent,
            llm_provider=lambda: None,   # 不调 LLM
            max_steps=1,
            enable_intent_router=False,
            shortlister_provider=None,
        )
        # 模拟 ask 模式：只有 safe 工具
        tools = agent.list_all_tools(allowed_danger_levels={"safe"})
        # 去掉 discovery 入口（ask 模式 allow_mcp_discovery=False）
        tools = {
            n: m for n, m in tools.items()
            if n not in AgentOrchestrator._MCP_DISCOVERY_TOOLS
        }
        print(f"  输入 tools（ask 模式过滤后）count={len(tools)}")
        mcp_in = {n: m for n, m in tools.items() if n.startswith("mcp__")}
        print(f"  输入 MCP tools: {sorted(mcp_in.keys())}")

        picked = orch._select_shortlist(GOAL, tools)
        print(f"  shortlist count={len(picked)}")
        print(f"  shortlist: {sorted(picked.keys())}")
        tavily_picked = [n for n in picked if n.startswith("mcp__tavily")]
        fetch_picked = [n for n in picked if n.startswith("mcp__fetch")]
        if tavily_picked:
            _ok(f"tavily 在 shortlist 里: {tavily_picked}")
        else:
            _fail(
                "tavily 不在 shortlist 里 —— 问题在 _select_shortlist 的 MCP 分支"
                f"（fetch 是否在: {bool(fetch_picked)}）"
            )
    except Exception as exc:
        _fail(f"_select_shortlist 检查失败: {type(exc).__name__}: {exc}")

    # ── 5. search_intents 对这条 query 返回了什么 ──
    _line("5. search_intents(goal) 返回")
    try:
        from tools.mcp_intent_index import search_intents

        idx_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "mcp_intent_index",
        )
        print(f"  index dir: {idx_dir}")
        print(f"  faiss.index exists: {os.path.exists(os.path.join(idx_dir, 'faiss.index'))}")
        print(f"  records.json exists: {os.path.exists(os.path.join(idx_dir, 'records.json'))}")

        hits = search_intents(GOAL, top_k=15)
        if not hits:
            _warn("search_intents 返回空 —— 索引文件缺失 / query 改写失败 / 无命中")
        for h in hits:
            srv = h.get("server_name")
            print(f"\n  server={srv} top_score={h.get('top_score'):.4f}")
            for t in h.get("matched_tools", []):
                print(
                    f"    - {t.get('tool_name')} "
                    f"rrf={t.get('rrf_score'):.4f} "
                    f"text={str(t.get('text', ''))[:80]!r}"
                )
        tavily_hit = any(
            t.get("tool_name", "").startswith("mcp__tavily")
            for h in hits for t in h.get("matched_tools", [])
        )
        if tavily_hit:
            _ok("search_intents 命中了 tavily 工具")
        else:
            _fail(
                "search_intents 没命中 tavily —— 意图索引里 tavily 的合成意图"
                "可能不覆盖“历史事件/宕机/故障”。"
            )
    except Exception as exc:
        _fail(f"search_intents 检查失败: {type(exc).__name__}: {exc}")

    # ── 6. 索引 records 里 tavily 的文本 ──
    _line("6. records.json 里 tavily 的索引文本")
    try:
        records_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "mcp_intent_index", "records.json",
        )
        if not os.path.exists(records_path):
            _warn("records.json 不存在，跳过")
        else:
            with open(records_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            tavily_recs = [
                r for r in records
                if str(r.get("tool_name", "")).startswith("mcp__tavily")
            ]
            print(f"  tavily 索引记录数: {len(tavily_recs)}")
            for r in tavily_recs[:20]:
                print(f"    - kind={r.get('kind')} text={str(r.get('text', ''))[:100]!r}")
            if not tavily_recs:
                _fail("records.json 里没有 tavily 记录 —— 索引从未为 tavily 建过")
    except Exception as exc:
        _fail(f"records.json 检查失败: {type(exc).__name__}: {exc}")

    _line("诊断结束：看上面哪一层先 FAIL，就是 tavily 丢失的位置")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n中断")
        sys.exit(130