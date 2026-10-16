"""通用文本清洗工具。"""
import re


def clean_cn_spaces(text: str) -> str:
    """去掉中文之间的空格（如"钱 易"→"钱易"）。"""
    if not text:
        return text
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])', '', text)
        text = re.sub(r'(?<=[\u4e00-\u9fff]) (?=[，。！？、；：（）【】《》""''])', '', text)
        text = re.sub(r'(?<=[，。！？、；：（）【】《》""'']) (?=[\u4e00-\u9fff])', '', text)
    return text


def clean_heading(h: str) -> str:
    """清洗 heading：去项目符号 + 中文间空格。"""
    if not h:
        return h
    h = re.sub(r'^[\uf025\uf026\uf0d8\u2022\-\*\s]+', '', h).strip()
    return clean_cn_spaces(h)