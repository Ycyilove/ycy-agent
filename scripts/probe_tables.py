# scripts/probe_tables.py
import pymupdf
from pathlib import Path

pdf = Path("第十届“中国高校计算机大赛-团体程序设计天梯赛”全国总决赛获奖名单.pdf")
doc = pymupdf.open(pdf)
for page_no, page in enumerate(doc, start=1):
    tabs = page.find_tables()
    print(f"page {page_no}: {len(tabs.tables)} tables")
    for i, tab in enumerate(tabs.tables):
        rows = tab.extract()
        print(f"  table {i}: {len(rows)} rows x {len(rows[0]) if rows else 0} cols")
        if rows:
            print(f"    header: {rows[0][:5]}")
doc.close()