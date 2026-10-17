from backend.app import get_local_rag_service

rag = get_local_rag_service()
metadata = rag.vector_store.metadata

for TARGET in ["钱易", "汪澄", "朱剑枫", "管晏如", "张亲翰"]:
    found = []
    for i, m in enumerate(metadata):
        text = m.get("text", "")
        if TARGET in text:
            found.append((i, text))
    
    print(f"\n{'='*50}")
    print(f"含 {TARGET!r} 的 chunk: {len(found)} 个")
    for i, text in found[:2]:
        pos = text.find(TARGET)
        start = max(0, pos - 40)
        end = min(len(text), pos + len(TARGET) + 40)
        print(f"  [chunk {i}] ...{text[start:end]}...")