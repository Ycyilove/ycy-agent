"""
工具系统测试脚本
验证工具注册、意图识别、工具执行等功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.loader import load_all_tools
from tools.agent import ToolAgent


def test_tools_registration():
    """测试工具注册"""
    print("=" * 60)
    print("Test 1: Tool Registration")
    print("=" * 60)
    
    load_all_tools()
    agent = ToolAgent()
    
    tools = agent.registry.list_tools()
    print(f"[OK] Registered tools count: {len(tools)}")
    print(f"Tools list: {tools}\n")
    
    categories = {}
    for name, meta in agent.registry.get_all_metadata().items():
        cat = meta.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)
    
    print("[Categories]:")
    for cat, tools_list in categories.items():
        print(f"  [{cat}]: {', '.join(tools_list)}")
    print()


def test_intent_recognition():
    """测试意图识别"""
    print("=" * 60)
    print("Test 2: Intent Recognition")
    print("=" * 60)
    
    load_all_tools()
    agent = ToolAgent()
    
    test_messages = [
        "运行Python代码 print('hello world')",
        "读取CSV文件 data.csv",
        "分析这个excel文件",
        "现在几点了",
        "计算2024-01-01到2024-12-31相差多少天",
        "重命名文件 old.txt 为 new.txt",
        "把data.json转成csv格式",
        "获取 https://api.github.com 的内容",
    ]
    
    for msg in test_messages:
        intent = agent.analyze_intent(msg)
        print(f"[Input]: {msg}")
        print(f"   Intent type: {intent.intent_type}")
        print(f"   Needs tool: {intent.needs_tool}")
        if intent.suggested_tool:
            print(f"   Suggested tool: {intent.suggested_tool}")
            print(f"   Parameters: {intent.parameters}")
        print()


def test_sandbox_tool():
    """测试沙箱工具"""
    print("=" * 60)
    print("Test 3: Sandbox Tool Execution")
    print("=" * 60)
    
    from tools.sandbox import run_python_code
    
    test_cases = [
        ("print(1+2+3)", "Simple calculation"),
        ("result = [x**2 for x in range(10)]; print(sum(result))", "List comprehension"),
        ("import os", "Dangerous operation - should be blocked"),
    ]
    
    for code, desc in test_cases:
        print(f"[Test]: {desc}")
        print(f"   Code: {code}")
        result = run_python_code(code)
        print(f"   Success: {result.success}")
        if result.success:
            print(f"   Output: {result.output}")
        else:
            print(f"   Error: {result.error}")
        print()


def test_time_tools():
    """测试时间工具"""
    print("=" * 60)
    print("Test 4: Time Tools")
    print("=" * 60)
    
    from tools.time_tools import get_current_time, calculate_time_difference, calculate_age
    
    print("[Current Time]:")
    result = get_current_time()
    print(f"   {result.get('formatted')}")
    
    print("\n[Time Difference] (2024-01-01 to 2024-01-15):")
    result = calculate_time_difference("2024-01-01", "2024-01-15")
    if result.get('success'):
        print(f"   {result.get('difference')}")
    
    print("\n[Age Calculation] (1990-01-01):")
    result = calculate_age("1990-01-01")
    if result.get('success'):
        print(f"   Age: {result.get('age')} years old")
    print()


def test_file_tools():
    """测试文件工具"""
    print("=" * 60)
    print("Test 5: File Tools")
    print("=" * 60)
    
    from tools.file_tools import list_files, get_file_info
    
    print("[List Files in Current Directory]:")
    result = list_files(".")
    if result.get('success'):
        print(f"   Files count: {result.get('files_count')}")
        print(f"   Dirs count: {result.get('dirs_count')}")
        print(f"   First 5 files: {[f['name'] for f in result.get('files', [])[:5]]}")
    print()


def test_network_tools():
    """测试网络工具"""
    print("=" * 60)
    print("Test 6: Network Tools")
    print("=" * 60)
    
    from tools.network_tools import check_url_status
    
    print("[Check Baidu Website Status]:")
    result = check_url_status("https://www.baidu.com")
    print(f"   Success: {result.get('success')}")
    if result.get('success'):
        print(f"   Status code: {result.get('status_code')}")
        print(f"   Response time: {result.get('response_time_ms')}ms")
    print()


def test_tool_execution():
    """测试完整工具执行流程"""
    print("=" * 60)
    print("Test 7: Complete Tool Execution Flow")
    print("=" * 60)
    
    load_all_tools()
    agent = ToolAgent()
    
    message = "计算 1+2+3+4+5 的结果"
    print(f"[User Input]: {message}")
    
    intent = agent.analyze_intent(message)
    print(f"   Recognized intent: {intent.intent_type}")
    
    if intent.needs_tool:
        tool_call = agent.prepare_tool_call(intent)
        print(f"   Prepare to call tool: {tool_call.tool_name}")
        print(f"   Parameters: {tool_call.parameters}")
        
        result = agent.execute_tool(tool_call)
        formatted = agent.format_tool_result(result)
        print(f"\n[Execution Result]:\n{formatted}")
    print()


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("\n[Tool System Test]\n")
    
    test_tools_registration()
    test_intent_recognition()
    test_sandbox_tool()
    test_time_tools()
    test_file_tools()
    test_network_tools()
    test_tool_execution()
    
    print("=" * 60)
    print("[All tests completed!]")
    print("=" * 60)
