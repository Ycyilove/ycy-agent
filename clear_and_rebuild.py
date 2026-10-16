# -*- coding: utf-8 -*-
"""清空知识库并重建索引。

用途：
    修改分块策略 / 嵌入模型 / 文档树解析后，需要清空旧索引重新入库。

用法：
    python clear_and_rebuild.py                    # 只清空，不上传
    python clear_and_rebuild.py <文件或目录> [...]   # 清空 + 上传
    python clear_and_rebuild.py --yes data/        # 跳过确认，直接清空 + 上传
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import get_local_rag_service  # noqa: E402


SUPPORTED_EXTS = {'.pdf', '.txt', '.docx'}


def collect_files(paths):
    """展开文件 / 目录为文件列表。"""
    files = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"⚠️ 路径不存在，跳过: {p}")
            continue
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTS:
                files.append(path)
            else:
                print(f"⚠️ 不支持的文件类型，跳过: {p}")
        elif path.is_dir():
            for child in sorted(path.rglob('*')):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTS:
                    files.append(child)
    return files


def confirm(prompt: str, skip: bool = False) -> bool:
    """确认操作。"""
    if skip:
        return True
    ans = input(f"{prompt} [y/N]: ").strip().lower()
    return ans in ('y', 'yes')


def do_clear(rag):
    """清空知识库。"""
    print("\n" + "=" * 60)
    print("清空知识库")
    print("=" * 60)

    before = rag.get_stats()
    print(f"清空前: {before.get('total_files')} 个文件, "
          f"{before.get('total_chunks')} 个 chunk")

    result = rag.clear_all()
    print(f"clear_all 返回: {result}")

    after = rag.get_stats()
    print(f"清空后: {after.get('total_files')} 个文件, "
          f"{after.get('total_chunks')} 个 chunk")

    # 校验
    if hasattr(rag.vector_store, 'verify_consistency'):
        print(f"一致性: {rag.vector_store.verify_consistency()}")

    return after.get('total_chunks', 0) == 0


def do_upload(rag, files):
    """上传文件。"""
    print("\n" + "=" * 60)
    print(f"上传 {len(files)} 个文件")
    print("=" * 60)

    ok, fail = 0, 0
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {file_path.name}")
        try:
            content = file_path.read_bytes()
        except OSError as e:
            print(f"  ❌ 读取失败: {e}")
            fail += 1
            continue

        if not content:
            print(f"  ❌ 文件内容为空")
            fail += 1
            continue

        t0 = time.time()
        try:
            result = rag.add_file(content, file_path.name)
        except Exception as e:
            print(f"  ❌ 处理异常: {e}")
            fail += 1
            continue
        elapsed = time.time() - t0

        status = result.get('status', 'unknown')
        message = result.get('message', '')
        if status == 'success':
            print(f"  ✅ {message} ({elapsed:.2f}s)")
            ok += 1
        elif status == 'warning':
            print(f"  ⚠️ {message} ({elapsed:.2f}s)")
            fail += 1
        else:
            print(f"  ❌ {message} ({elapsed:.2f}s)")
            fail += 1

    return ok, fail


def print_stats(rag):
    """打印最终统计。"""
    stats = rag.get_stats()
    print("\n" + "=" * 60)
    print("最终统计")
    print("=" * 60)
    print(f"embedding_model: {stats.get('embedding_model')}")
    print(f"total_files:     {stats.get('total_files')}")
    print(f"total_chunks:    {stats.get('total_chunks')}")
    for f in stats.get('files', []):
        print(f"  - {f['filename']}: {f['chunk_count']} chunks")

    if hasattr(rag.vector_store, 'verify_consistency'):
        print(f"consistency:     {rag.vector_store.verify_consistency()}")


def main():
    args = sys.argv[1:]

    skip_confirm = False
    if '--yes' in args:
        skip_confirm = True
        args.remove('--yes')
    if '-y' in args:
        skip_confirm = True
        args.remove('-y')

    files = collect_files(args) if args else []

    print("=" * 60)
    print("清空重建知识库")
    print("=" * 60)
    if files:
        print(f"待上传文件: {len(files)} 个")
        for f in files:
            print(f"  - {f}")
    else:
        print("未指定文件，仅执行清空。")

    # 初始化 RAG 服务
    print("\n初始化 RAG 服务（首次加载模型可能较慢）...")
    rag = get_local_rag_service()

    # 确认
    if not confirm("\n⚠️ 将清空知识库，确认继续？", skip_confirm):
        print("已取消。")
        return

    # 清空
    cleared = do_clear(rag)
    if not cleared:
        print("❌ 清空失败，索引可能不一致，中止。")
        sys.exit(1)

    # 上传
    if files:
        ok, fail = do_upload(rag, files)
        print(f"\n上传完成: 成功 {ok}, 失败 {fail}")

    # 统计
    print_stats(rag)


if __name__ == '__main__':
    main()

'''
# 只清空（交互确认）
python clear_and_rebuild.py

# 清空 + 上传单个文件
python clear_and_rebuild.py "获奖名单.pdf"

# 清空 + 上传目录
python clear_and_rebuild.py data/

# 清空 + 上传多个，跳过确认
python clear_and_rebuild.py --yes data/ notes.txt

# 单文件 + 跳过确认
python clear_and_rebuild.py -y "获奖名单.pdf"
'''