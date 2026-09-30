from qdrant_client import QdrantClient

client = QdrantClient(path=r"D:\ycy\LLM\LLM\data\mem0_qdrant")

# 列出所有 collection
print("Collections:", client.get_collections())

# 查看 agent_facts 的所有点
points = client.scroll(
    collection_name="agent_facts",
    limit=100,
    with_payload=True,
    with_vectors=False,
)
records = points[0]
print(f"\nTotal points: {len(records)}\n")
for i, p in enumerate(records, 1):
    print(f"--- [{i}] ---")
    print(p.payload)
    print()