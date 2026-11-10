"""
统一结构纠正器：基于通用的排版语法和启发式规则，重建文本层级。
坚决不硬编码具体业务实体，所有领域规则均通过配置注入。
"""
import re
import logging
from typing import List, Any, Optional, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class StructureRule:
    """结构规则：匹配到 pattern 即赋予对应的 level"""
    pattern: str
    level: int
    description: str = ""

class HierarchyCorrector:
    def __init__(self, domain_rules: Optional[List[StructureRule]] = None):
        # 1. 内置通用排版规则（跨领域、跨语言通用，严禁包含具体业务实体）
        self.syntax_rules = [
            # 中文编号：一、 二、 三、
            StructureRule(pattern=r"^[一二三四五六七八九十]+、", level=1, description="中文大写数字编号"),
            # 中文括号编号：（一） （二）
            StructureRule(pattern=r"^（[一二三四五六七八九十]+）", level=2, description="中文括号数字编号"),
            # 阿拉伯数字编号：1. 2. 
            StructureRule(pattern=r"^\d+\.\s+[^\d]", level=1, description="阿拉伯数字编号"),
            # 多级阿拉伯数字编号：1.1 1.1.1 (动态计算层级)
            StructureRule(pattern=r"^(\d+(?:\.\d+)+)\s+", level=-1, description="多级数字编号"),
            # 英文常见章节
            StructureRule(pattern=r"^(Chapter|Section|Part)\s+\d+", level=1, description="英文章节"),
            # 列表符号：-, *, •
            StructureRule(pattern=r"^[\-\*•]\s+", level=4, description="无序列表符"),
            # 纯英文大写或首字母大写的短句（通常为标题）
            StructureRule(pattern=r"^[A-Z][A-Za-z0-9\s]{2,30}$", level=2, description="英文短标题"),
        ]
        
        # 2. 注入的外部领域规则（如来自 field_schema.json 的 header_hints）
        self.domain_rules = domain_rules or []
        
    def correct_chunks(self, chunks: List[Any]) -> List[Any]:
        """遍历所有 chunk，根据文本和上下文修正 meta.heading_levels。"""
        if not chunks:
            return chunks

        # 维护一个当前活动的标题栈：[(text, level), ...]
        active_stack = []

        for chunk in chunks:
            headings = getattr(chunk.meta, "headings", None) or []
            if not headings:
                continue

            # 如果 chunk 文本很长，说明它包含了正文，可以利用正文长度反推标题
            chunk_text = getattr(chunk, "text", "") or ""
            chunk_len = len(chunk_text.strip())

            corrected_levels = []
            
            for idx, h in enumerate(headings):
                h_clean = h.strip()
                
                # 通过多重信号推断层级
                inferred_level = self._infer_level(
                    h_clean, active_stack, is_last_heading=(idx == len(headings) - 1), chunk_len=chunk_len
                )
                
                # 防止层级跳跃过大（例如从 H1 直接跳到 H4，限制一次最多降 1 级）
                if active_stack and inferred_level > active_stack[-1][1] + 1:
                    inferred_level = active_stack[-1][1] + 1
                
                # 更新栈：弹出比当前级别深或相等的旧标题
                while active_stack and active_stack[-1][1] >= inferred_level:
                    active_stack.pop()
                active_stack.append((h_clean, inferred_level))
                
                corrected_levels.append(inferred_level)

            # 将纠正后的 levels 写回
            try:
                chunk.meta.heading_levels = corrected_levels
            except (AttributeError, ValueError):
                pass # 兼容只读的 Pydantic 模型
                
        return chunks

    def _infer_level(self, text: str, active_stack: List[tuple], is_last_heading: bool, chunk_len: int) -> int:
        """综合多重启发式信号推断单行文本的层级"""
        
        # --- 信号 1：显式语法和领域规则匹配 ---
        for rule in self.syntax_rules + self.domain_rules:
            m = re.match(rule.pattern, text, re.IGNORECASE)
            if m:
                if rule.level == -1 and m.groups():
                    # 动态计算数字层级：如 1.1.1 -> count('.') + 1 = 3
                    return m.group(1).count(".") + 1
                return rule.level

        # --- 信号 2：基于上下文和文本形态的启发式推断 ---
        if active_stack:
            last_text, last_level = active_stack[-1]
            
            # 2.1 如果文本与栈顶完全一样，保持同级
            if text == last_text:
                return last_level

            # 2.2 标点符号分析：标题通常不包含句号、问号等结束符
            has_ending_punct = bool(re.search(r"[。！？.!?]$", text))
            has_comma = bool(re.search(r"[，,；;]", text))
            
            # 2.3 长度分析：标题一般较短（少于 40 字符）
            is_short = len(text) < 40
            
            # 2.4 文本长度反推：如果当前 chunk 包含大量正文（>100字），
            # 那么它前面的 headings 列表里最后一个标题，很可能是真正的标题。
            # 反之，如果 chunk 极短（纯标题集合），层级应该更深。
            if is_last_heading and chunk_len > 100 and is_short and not has_ending_punct:
                # 这是一个典型的“紧跟长段落的标题”，保持其原本深度
                return last_level
            
            # 2.5 如果没有标点、而且很短，大概率是未识别出的子标题（层级加深）
            if is_short and not has_ending_punct and not has_comma:
                # 避免无限制加深，最多加深一级
                return min(last_level + 1, 6)
                
        # --- 兜底 ---
        # 如果都判断不了，默认给 1 级（通常对应文档标题）
        return 1