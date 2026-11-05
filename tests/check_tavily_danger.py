"""确认 tavily 工具的 danger_level / origin，以及 ask 模式 safe 过滤后是否还在。"""
import asyncio
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import backend.config  # noqa: F401


async def main():
    from mcp_client.config import load_mcp_config
    from mcp_client.manager import MCPClientManager
    from tools.agent import ToolAgent

    configs = load_mcp_config()
    print("configs:", [c.name for c in configs])

    mg = MCPClientManager(configs)
    agent = ToolAgent()
    agent.set_mcp_client_manager(mg)

    # 先看看 discover 之前 registry 里有没有 tavily
    meta_before = agent.registry.get_all_metadata()
    print("before discover, tavily in registry:",
          [n for n in meta_before if n.startswith("mcp__tavily")])

    print("\nconnect + discover tavily ...")
    await mg.connect("tavily")
    await mg.discover("tavily", agent.register_dynamic_tool,
                      agent.unregister_dynamic_tool)
    print("tavily status:", mg.statuses.get("tavily"))

    # 1. registry 里 tavily 的 danger_level / origin
    print("\n--- registry: tavily tools ---")
    meta = agent.registry.get_all_metadata()
    tavily_meta = {n: m for n, m in meta.items() if n.startswith("mcp__tavily")}
    if not tavily_meta:
        print("  [FAIL] registry 里没有 mcp__tavily__*")
    for n, m in tavily_meta.items():
        print(f"  {n}: danger_level={m.danger_level!r} origin={m.origin!r} "
              f"idempotent={getattr(m, 'idempotent', None)!r}")

    # 2. ask 模式 list_all_tools({'safe'}) 是否包含 tavily
    print("\n--- list_all_tools ---")
    tools_all = agent.list_all_tools(allowed_danger_levels=None)
    tools_safe = agent.list_all_tools(allowed_danger_levels={"safe"})
    print("  all  count:", len(tools_all),
          " tavily:", [n for n in tools_all if n.startswith("mcp__tavily")])
    print("  safe count:", len(tools_safe),
          " tavily:", [n for n in tools_safe if n.startswith("mcp__tavily")],
          " fetch:", [n for n in tools_safe if n.startswith("mcp__fetch")])

    if tavily_meta and not [n for n in tools_safe if n.startswith("mcp__tavily")]:
        print("\n  >>> 根因：tavily danger_level 不是 safe，ask 模式过滤掉了它")
    elif not tavily_meta:
        print("\n  >>> 根因：discover 没把 tavily 注册进 registry")
    else:
        print("\n  >>> tavily 在 safe 工具集里，ask 模式能看到它")

    # 3. search_intents 对这条 goal 返回什么
    print("\n--- search_intents ---")
    try:
        from tools.mcp_intent_index import search_intents
        hits = search_intents(
            "我想知道deepseek曾经出现过的所有服务器繁忙的宕机事件", top_k=15,
        )
        if not hits:
            print("  [WARN] 返回空")
        for h in hits:
            print(f"  server={h['server_name']} top_score={h.get('top_score')}")
            for t in h.get("matched_tools", []):
                print(f"    - {t.get('tool_name')} rrf={t.get('rrf_score')} "
                      f"text={str(t.get('text',''))[:60]!r}")
        tavily_hit = any(
            t.get("tool_name", "").startswith("mcp__tavily")
            for h in hits for t in h.get("matched_tools", [])
        )
        print("  tavily hit:", tavily_hit)
    except Exception as exc:
        print(f"  search_intents 失败: {type(exc).__name__}: {exc}")

    await mg.close()


if __name__ == "__main__":
    asyncio.run(main())