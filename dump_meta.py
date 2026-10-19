# -*- coding: utf-8 -*-
import pickle
import sys
from pathlib import Path

# Windows 终端默认 GBK，强制 UTF-8，避免打印中文报 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

pkl = Path("vector_store/metadata.pkl")
print("pkl path:", pkl.resolve())
print("pkl exists:", pkl.exists())

with pkl.open("rb") as f:
    meta = pickle.load(f)

print("total:", len(meta))
if meta:
    print("keys of first:", sorted(meta[0].keys()))

hits = []
for i, m in enumerate(meta):
    t = m.get("text", "") or ""
    o = m.get("original_text", "") or ""
    if "杨灿垣" in t or "杨灿垣" in o:
        hits.append((i, m))

print("hits:", len(hits))

out_path = Path("dump_meta.txt")
with out_path.open("w", encoding="utf-8") as out:
    def w(*args):
        line = " ".join(str(a) for a in args)
        print(line)
        out.write(line + "\n")

    w("total:", len(meta))
    w("hits:", len(hits))
    for i, m in hits:
        w("=" * 60)
        w("index:", i)
        w("chunk_id:", m.get("chunk_id"))
        w("headings:", m.get("headings"))
        w("has_contextual:", m.get("has_contextual"))
        text = m.get("text", "") or ""
        original = m.get("original_text", "") or ""
        w("text_len:", len(text), "original_len:", len(original))
        w("--- text[:2000] ---")
        w(text[:2000])
        w("--- original[:2000] ---")
        w(original[:2000])

print("written to:", out_path.resolve())