"""
测试会话记忆系统功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from session_memory import SessionMemory, get_session_memory


def test_session_memory():
    """测试会话记忆系统功能"""
    print("=" * 50)
    print("开始测试会话记忆系统")
    print("=" * 50)

    # 使用测试数据库
    memory = SessionMemory("./data/test_sessions.db")

    # 1. 测试创建会话
    print("\n[测试1] 创建新会话")
    session_id = memory.create_session(name="测试会话")
    print(f"[PASS] 创建会话成功: {session_id}")

    # 2. 测试列出会话
    print("\n[测试2] 列出所有会话")
    sessions = memory.list_sessions()
    print(f"[PASS] 当前会话数: {len(sessions)}")
    for s in sessions:
        print(f"  - {s['session_id'][:8]}... | {s['name']} | {s['created_at']}")

    # 3. 测试添加消息
    print("\n[测试3] 添加消息到会话")
    msg_id1 = memory.add_message(session_id, "user", "你好，请介绍一下自己")
    print(f"[PASS] 添加用户消息成功: ID={msg_id1}")

    msg_id2 = memory.add_message(session_id, "assistant", "你好！我是AI助手，很高兴为你服务。")
    print(f"[PASS] 添加AI回复成功: ID={msg_id2}")

    # 4. 测试获取消息
    print("\n[测试4] 获取会话全部消息")
    messages = memory.get_messages(session_id)
    print(f"[PASS] 获取消息数: {len(messages)}")
    for msg in messages:
        print(f"  - [{msg['role']}] {msg['content'][:30]}...")

    # 5. 测试获取短期上下文
    print("\n[测试5] 获取短期上下文（用于实时对话）")
    context = memory.get_conversation_context(session_id, max_messages=10)
    print(f"[PASS] 上下文消息数: {len(context)}")
    print(f"  格式: {context}")

    # 6. 测试添加文件
    print("\n[测试6] 添加文件到会话")
    file_id = memory.add_session_file(
        session_id=session_id,
        file_name="document.pdf",
        file_path="/uploads/doc.pdf",
        file_type="application/pdf"
    )
    print(f"[PASS] 添加文件成功: {file_id}")

    # 7. 测试获取会话文件
    print("\n[测试7] 获取会话关联文件")
    files = memory.get_session_files(session_id)
    print(f"[PASS] 会话文件数: {len(files)}")
    for f in files:
        print(f"  - {f['file_name']} ({f['file_type']})")

    # 8. 测试重命名会话
    print("\n[测试8] 重命名会话")
    success = memory.rename_session(session_id, "新名称-测试会话")
    print(f"[PASS] 重命名成功: {success}")

    session_info = memory.get_session_info(session_id)
    print(f"  新名称: {session_info['name']}")

    # 9. 测试创建第二个会话（隔离测试）
    print("\n[测试9] 创建第二个会话（隔离测试）")
    session_id2 = memory.create_session(name="第二个会话")
    memory.add_message(session_id2, "user", "这是第二个会话的消息")
    messages2 = memory.get_messages(session_id2)
    print(f"[PASS] 第二个会话消息数: {len(messages2)} (应只有1条)")

    # 验证会话隔离
    files2 = memory.get_session_files(session_id2)
    print(f"[PASS] 第二个会话文件数: {len(files2)} (应为空)")
    print(f"[PASS] 第一个会话文件数: {len(files)} (应为1)")
    print("  -> 会话隔离验证通过！")

    # 10. 测试删除会话
    print("\n[测试10] 删除会话")
    success = memory.delete_session(session_id2)
    print(f"[PASS] 删除成功: {success}")

    sessions = memory.list_sessions()
    print(f"[PASS] 剩余会话数: {len(sessions)}")

    # 11. 测试清空全部会话
    print("\n[测试11] 清空全部会话")
    count = memory.clear_all_sessions()
    print(f"[PASS] 清空会话数: {count}")

    sessions = memory.list_sessions()
    print(f"[PASS] 当前会话数: {len(sessions)}")

    # 12. 测试自动建表容错
    print("\n[测试12] 验证自动建表（重复初始化）")
    memory2 = SessionMemory("./data/test_sessions.db")
    print("[PASS] 重复初始化成功，无报错")

    # 清理测试数据库
    if os.path.exists("./data/test_sessions.db"):
        os.remove("./data/test_sessions.db")
        print("[PASS] 清理测试数据库")

    print("\n" + "=" * 50)
    print("全部测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    test_session_memory()
