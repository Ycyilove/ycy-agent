# -*- coding: utf-8 -*-
"""
NLP 预处理模块
提供中文分词、词性标注、关键词提取等功能，用于增强意图识别的准确性
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

try:
    import jieba
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    jieba = None


@dataclass
class TokenInfo:
    """分词结果信息"""
    word: str
    flag: str
    position: int


class NLPProcessor:
    """
    NLP 预处理器
    负责中文分词、词性标注、关键信息提取等任务
    """

    def __init__(self, user_dict: Dict[str, str] = None):
        """
        初始化 NLP 处理器

        Args:
            user_dict: 用户自定义词典，格式: {词: 词性}
        """
        self.user_dict = user_dict or {}
        self._init_jieba()

    def _init_jieba(self):
        """初始化 jieba 分词器，添加自定义词典"""
        if not JIEBA_AVAILABLE:
            return

        # 添加自定义词典，提高专业术语识别率
        custom_words = [
            ("CSV", "eng", 100),
            ("JSON", "eng", 100),
            ("Excel", "eng", 100),
            ("Python", "eng", 100),
            ("API", "eng", 100),
            ("URL", "eng", 100),
            ("HTTP", "eng", 100),
            ("出生日期", "n", 100),
            ("计算年龄", "v", 100),
            ("导出文件", "v", 100),
            ("读取文件", "v", 100),
        ]

        for word, flag, freq in custom_words:
            if jieba:
                jieba.add_word(word, freq, flag)

        # 添加用户自定义词典
        for word, flag in self.user_dict.items():
            if jieba:
                jieba.add_word(word, 100, flag)

    def tokenize(self, text: str) -> List[str]:
        """
        分词处理

        Args:
            text: 输入文本

        Returns:
            分词后的词语列表
        """
        if not JIEBA_AVAILABLE or not text:
            return text.split()

        # 精确模式分词
        return list(jieba.cut(text, cut_all=False))

    def tokenize_with_pos(self, text: str) -> List[TokenInfo]:
        """
        分词并标注词性

        Args:
            text: 输入文本

        Returns:
            分词结果列表，每项包含词语、词性和位置
        """
        if not JIEBA_AVAILABLE or not text:
            return []

        tokens = []
        words = pseg.cut(text)

        position = 0
        for word, flag in words:
            tokens.append(TokenInfo(
                word=word,
                flag=flag,
                position=position
            ))
            position += len(word)

        return tokens

    def extract_numbers(self, text: str) -> List[Tuple[str, int]]:
        """
        提取文本中的数字及其位置

        Args:
            text: 输入文本

        Returns:
            数字列表及其在文本中的位置
        """
        pattern = r'\d+\.?\d*'
        matches = re.finditer(pattern, text)
        return [(m.group(), m.start()) for m in matches]

    def extract_dates(self, text: str) -> List[Tuple[str, int]]:
        """
        提取文本中的日期信息

        Args:
            text: 输入文本

        Returns:
            日期列表及其在文本中的位置
        """
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # 2024-01-01 或 2024/01/01
            r'\d{4}年\d{1,2}月\d{1,2}日',   # 2024年1月1日
            r'\d{4}年\d{1,2}月',              # 2024年1月
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # 01-01-2024
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, text)
            dates.extend([(m.group(), m.start()) for m in matches])

        # 按位置排序
        dates.sort(key=lambda x: x[1])
        return dates

    def extract_urls(self, text: str) -> List[Tuple[str, int]]:
        """
        提取文本中的 URL

        Args:
            text: 输入文本

        Returns:
            URL 列表及其位置
        """
        pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        matches = re.finditer(pattern, text)
        return [(m.group(), m.start()) for m in matches]

    def extract_file_paths(self, text: str) -> List[Tuple[str, int]]:
        """
        提取文本中的文件路径

        Args:
            text: 输入文本

        Returns:
            文件路径列表及其位置
        """
        patterns = [
            r'["\']([^"\']+\.(csv|xlsx?|txt|json|xml))["\']',  # 引号内的路径
            r'(?:file|path)[:：]\s*([^\s]+)',                  # file: 或 path:
            r'([^\s]+\.(csv|xlsx?|txt|json|xml))',             # 无引号路径
        ]

        paths = []
        seen = set()
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for m in matches:
                path = m.group(1) if m.lastindex else m.group()
                if path not in seen:
                    paths.append((path, m.start()))
                    seen.add(path)

        paths.sort(key=lambda x: x[1])
        return paths

    def extract_json_like(self, text: str) -> List[Tuple[str, int]]:
        """
        提取文本中类似 JSON 的数据结构

        Args:
            text: 输入文本

        Returns:
            JSON 字符串列表及其位置
        """
        patterns = [
            (r'\[[\s\S]*?\]', 0),  # 数组
            (r'\{[\s\S]*?\}', 0),  # 对象
        ]

        structures = []
        seen = set()
        for pattern, group_idx in patterns:
            matches = re.finditer(pattern, text)
            for m in matches:
                # 验证是否为有效的 JSON 结构
                content = m.group(group_idx)
                if self._is_valid_json_like(content) and content not in seen:
                    structures.append((content, m.start()))
                    seen.add(content)

        structures.sort(key=lambda x: x[1])
        return structures

    def _is_valid_json_like(self, text: str) -> bool:
        """
        检查文本是否是有效的 JSON 结构

        Args:
            text: 待检查文本

        Returns:
            是否有效
        """
        text = text.strip()
        if not text:
            return False

        # 基本检查：首尾字符匹配
        if text.startswith('[') and not text.endswith(']'):
            return False
        if text.startswith('{') and not text.endswith('}'):
            return False

        # 简单验证括号匹配
        brackets = {'[': 0, '{': 0}
        for char in text:
            if char == '[':
                brackets['['] += 1
            elif char == ']':
                brackets['['] -= 1
            elif char == '{':
                brackets['{'] += 1
            elif char == '}':
                brackets['{'] -= 1

        return brackets['['] == 0 and brackets['{'] == 0

    def extract_named_entities(self, text: str) -> List[Tuple[str, str, int]]:
        """
        提取命名实体（人名、地名、机构名等）

        Args:
            text: 输入文本

        Returns:
            实体列表，每项包含实体、类型和位置
        """
        if not JIEBA_AVAILABLE:
            return []

        entities = []
        words = pseg.cut(text)

        for word, flag in words:
            # nr: 人名, ns: 地名, nt: 机构名
            if flag in ['nr', 'ns', 'nt'] and len(word) > 1:
                entities.append((word, flag, text.find(word)))

        return entities

    def get_keyword_weights(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        提取关键词及其权重（基于 TF-IDF 思想）

        Args:
            text: 输入文本
            top_k: 返回的关键词数量

        Returns:
            关键词及其权重列表
        """
        if not JIEBA_AVAILABLE:
            return []

        import math

        # 统计词频
        words = self.tokenize(text)
        word_freq = {}
        for word in words:
            if len(word) > 1 and not word.isdigit():
                word_freq[word] = word_freq.get(word, 0) + 1

        # 计算权重（简化的 TF-IDF）
        total = sum(word_freq.values())
        weighted = []
        for word, freq in word_freq.items():
            # TF * 词长权重（长词权重更高）
            weight = (freq / total) * math.log(len(word) + 1)
            weighted.append((word, weight))

        # 排序并返回 top_k
        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted[:top_k]

    def preprocess_for_intent(self, text: str) -> Dict:
        """
        预处理文本用于意图识别

        Args:
            text: 输入文本

        Returns:
            预处理结果，包含分词、关键信息等
        """
        # 移除多余空白
        text = ' '.join(text.split())

        # 基本信息提取
        result = {
            'original': text,
            'tokens': self.tokenize(text),
            'numbers': self.extract_numbers(text),
            'dates': self.extract_dates(text),
            'urls': self.extract_urls(text),
            'file_paths': self.extract_file_paths(text),
            'json_structures': self.extract_json_like(text),
        }

        # 词性标注
        if JIEBA_AVAILABLE:
            tokens_with_pos = self.tokenize_with_pos(text)
            result['tokens_with_pos'] = [
                {'word': t.word, 'flag': t.flag, 'pos': t.position}
                for t in tokens_with_pos
            ]

            # 提取名词和动词（意图识别的关键）
            keywords = [t.word for t in tokens_with_pos if t.flag.startswith('n') or t.flag.startswith('v')]
            result['intent_keywords'] = keywords

        return result


# 全局 NLP 处理器实例（延迟初始化）
_nlp_processor: Optional[NLPProcessor] = None


def get_nlp_processor() -> NLPProcessor:
    """
    获取全局 NLP 处理器实例

    Returns:
        NLPProcessor 实例
    """
    global _nlp_processor
    if _nlp_processor is None:
        _nlp_processor = NLPProcessor()
    return _nlp_processor


def preprocess_intent_text(text: str) -> Dict:
    """
    便捷函数：对文本进行预处理用于意图识别

    Args:
        text: 输入文本

    Returns:
        预处理结果
    """
    processor = get_nlp_processor()
    return processor.preprocess_for_intent(text)