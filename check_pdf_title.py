"""排查 '全国团队二等奖' 到底在哪一页、是不是文字。"""
import sys
from pathlib import Path

PDF = "第十届“中国高校计算机大赛-团体程序设计天梯赛”全国总决赛获奖名单.pdf"

from pypdf import PdfReader
reader = PdfReader(PDF)
page = reader.pages[23]   # 第 24 页
text = page.extract_text()
print("=== 第 24 页文本 ===")
print(text[:1000])
