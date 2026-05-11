"""
Agent核心逻辑模块
实现意图识别、工具选择、参数提取、自主决策等功能
"""
import re
import json
import ast
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

from . import registry, get_registry, ToolMetadata

LLM_INTENT_PROMPT = """你是一个智能工具选择助手。根据用户消息和可用工具列表，选择最合适的工具并提取参数。

## 可用工具列表
{tool_descriptions}

## 用户消息
{user_message}

## 任务要求
请分析用户消息，判断是否需要调用工具，并提取相关参数。

## 输出格式（严格遵循JSON格式，不要添加任何额外解释）
{{
    "needs_tool": true或false,  // 是否需要调用工具
    "tool_name": "工具名称",      // 如果需要工具，给出工具的确切名称
    "parameters": {{              // 提取的参数，JSON对象格式
        "param_name": "param_value"
    }},
    "reasoning": "选择理由"       // 简要说明为什么选择或不选择工具
}}

## 关键判断规则
1. **只选择明确匹配的工具**：只有当用户消息明确表达了需要某种操作时，才返回 needs_tool: true
2. **闲聊/问候不需要工具**：如"你好"、"今天天气怎么样"等属于闲聊，不触发工具
3. **参数提取要求**：
   - 日期格式：优先使用 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:mm:ss" 格式
   - JSON数据：直接提取用户提供的JSON结构，不要修改
   - 文件路径：提取用户提到的文件路径，如果用户没有指定路径，设置空字符串 ""
   - 数字和计算：提取相关数值
4. **仅使用列表中的工具**：不要臆造工具名称
5. **避免过度解读**：如果用户只是随口一说，不要强行关联到工具

## 常见场景判断
- "计算1+2+3" → 需要工具（run_python_code）
- "年龄多大" → 需要工具（calculate_age），提取birth_date
- "导出数据到CSV" → 需要工具（export_to_csv），提取data和file_path
- "帮我看看这个网页" → 需要工具（http_get），提取url
- "今天周三了吗" → 不需要工具（闲聊/一般性问题）
- "我想查天气" → 不需要工具（没有具体城市信息）"""


class LLMIntentRecognizer:
    """基于LLM的意图识别器"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._tool_descriptions_cache = None

    def _get_llm_client(self):
        """获取LLM客户端"""
        if self.llm_client:
            return self.llm_client
        try:
            from langchain_service import get_llm
            return get_llm()
        except ImportError:
            return None

    def _get_tool_descriptions(self, registry_instance) -> str:
        """获取工具描述列表"""
        if self._tool_descriptions_cache is None:
            metadata_dict = registry_instance.get_all_metadata()
            descriptions = []
            for tool_name, tool in metadata_dict.items():
                param_str = ", ".join([
                    f"{name}: {info.get('type', 'str')}"
                    for name, info in (tool.parameters or {}).items()
                ])
                descriptions.append(
                    f"- {tool.name}: {tool.description} (参数: {param_str})"
                )
            self._tool_descriptions_cache = "\n".join(descriptions)
        return self._tool_descriptions_cache

    def analyze(self, user_message: str, registry_instance=None) -> IntentAnalysis:
        """使用LLM分析用户意图"""
        if registry_instance is None:
            registry_instance = get_registry()

        llm = self._get_llm_client()
        if not llm:
            return IntentAnalysis(
                needs_tool=False,
                intent_type="llm_unavailable",
                confidence=0.0,
                reasoning="LLM不可用，无法进行智能意图分析"
            )

        tool_descriptions = self._get_tool_descriptions(registry_instance)
        prompt = LLM_INTENT_PROMPT.format(
            tool_descriptions=tool_descriptions,
            user_message=user_message
        )

        try:
            response = llm.generate([{"role": "user", "content": prompt}])

            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()

            result = json.loads(response_clean)

            needs_tool = result.get("needs_tool", False)
            tool_name = result.get("tool_name")
            parameters = result.get("parameters", {})
            reasoning = result.get("reasoning", "")

            if not isinstance(parameters, dict):
                parameters = {}

            intent_type = "llm_"
            if tool_name:
                intent_type += tool_name
            else:
                intent_type += "conversation"

            return IntentAnalysis(
                needs_tool=needs_tool,
                intent_type=intent_type,
                confidence=0.95,
                suggested_tool=tool_name,
                parameters=parameters,
                reasoning=reasoning
            )

        except json.JSONDecodeError as e:
            return IntentAnalysis(
                needs_tool=False,
                intent_type="parse_error",
                confidence=0.0,
                reasoning=f"LLM返回格式错误: {str(e)}"
            )
        except Exception as e:
            return IntentAnalysis(
                needs_tool=False,
                intent_type="error",
                confidence=0.0,
                reasoning=f"LLM分析失败: {str(e)}"
            )


@dataclass
class ToolCallRequest:
    """工具调用请求"""
    tool_name: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    user_confirmation_needed: bool


@dataclass
class ToolCallResult:
    """工具调用结果"""
    tool_name: str
    success: bool
    result: Any = None
    error: str = None
    execution_time: float = 0.0


@dataclass
class IntentAnalysis:
    """意图分析结果"""
    needs_tool: bool
    intent_type: str
    confidence: float
    suggested_tool: str = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class IntentRecognizer:
    """
    意图识别器 - 基于规则模式匹配 + NLP 预处理
    结合中文分词和关键词匹配，提供更准确的意图识别能力
    """

    def __init__(self):
        # 意图模式定义，包含关键词、工具名、优先级和参数提取器
        self.intent_patterns = {
            "export_to_csv": {
                "keywords": ["导出", "导出csv", "导出为csv", "导出到csv", "保存csv", "输出csv", "export csv", "to csv", "转为csv", "保存为csv"],
                "tool": "export_to_csv",
                "priority": 10,
                "param_extractors": {
                    "data": self._extract_json_data,
                    "file_path": self._extract_export_file_path,
                    "columns": self._extract_columns
                }
            },
            "run_code": {
                "keywords": ["运行代码", "执行代码", "跑一下", "run code", "execute", "python", "计算", "算一下"],
                "tool": "run_python_code",
                "priority": 5,
                "param_extractors": {
                    "code": self._extract_code_block
                }
            },
            "read_csv": {
                "keywords": ["读取csv", "读csv", "打开csv", "read csv", "load csv", "查看csv"],
                "tool": "read_csv",
                "priority": 5,
                "param_extractors": {
                    "file_path": self._extract_file_path
                }
            },
            "analyze_csv": {
                "keywords": ["分析csv", "分析数据", "统计", "分析这份", "数据分析", "analyze", "statistics", "分析一下"],
                "tool": "analyze_csv",
                "param_extractors": {
                    "file_path": self._extract_file_path
                }
            },
            "read_excel": {
                "keywords": ["读取excel", "读excel", "打开excel", "excel文件", "read excel", "load excel", "xlsx"],
                "tool": "read_excel",
                "param_extractors": {
                    "file_path": self._extract_file_path
                }
            },
            "list_files": {
                "keywords": ["列出文件", "查看文件", "目录", "文件列表", "list files", "ls", "dir"],
                "tool": "list_files",
                "param_extractors": {
                    "directory": self._extract_directory
                }
            },
            "rename_file": {
                "keywords": ["重命名", "改名", "rename", "改名为"],
                "tool": "rename_file",
                "param_extractors": {
                    "old_path": self._extract_old_path,
                    "new_name": self._extract_new_name
                }
            },
            "convert_format": {
                "keywords": ["转换格式", "格式转换", "convert", "转成", "转为"],
                "tool": "convert_file_format",
                "param_extractors": {
                    "file_path": self._extract_file_path,
                    "target_format": self._extract_format
                }
            },
            "get_file_info": {
                "keywords": ["文件信息", "文件详情", "文件大小", "file info", "文件哈希"],
                "tool": "get_file_info",
                "param_extractors": {
                    "file_path": self._extract_file_path
                }
            },
            "time_difference": {
                "keywords": ["时间差", "相隔", "间隔", "多少天", "时间间隔", "days between", "time difference"],
                "tool": "calculate_time_difference",
                "param_extractors": {
                    "start_time": self._extract_start_time,
                    "end_time": self._extract_end_time
                }
            },
            "add_time": {
                "keywords": ["加", "减去", "天后", "天后", "时间推移", "add time", "later", "after"],
                "tool": "add_time",
                "param_extractors": {
                    "base_time": self._extract_time,
                    "value": self._extract_time_value,
                    "unit": self._extract_time_unit
                }
            },
            "current_time": {
                "keywords": ["当前时间", "现在几点", "时间", "current time", "now", "几点"],
                "tool": "get_current_time",
                "priority": 3,
                "param_extractors": {}
            },
            "format_timestamp": {
                "keywords": ["时间格式", "格式化时间", "timestamp", "转换日期"],
                "tool": "format_timestamp",
                "param_extractors": {
                    "input_value": self._extract_time,
                    "output_format": self._extract_format
                }
            },
            "calculate_age": {
                "keywords": ["年龄", "几岁", "多大了", "age", "岁数", "多大"],
                "tool": "calculate_age",
                "priority": 5,
                "param_extractors": {
                    "birth_date": self._extract_birth_date
                }
            },
            "http_get": {
                "keywords": ["获取", "请求", "fetch", "get", "访问", "查看网页"],
                "tool": "http_get",
                "priority": 4,
                "param_extractors": {
                    "url": self._extract_url
                }
            },
            "http_post": {
                "keywords": ["提交", "post", "发送数据"],
                "tool": "http_post",
                "param_extractors": {
                    "url": self._extract_url,
                    "data": self._extract_post_data
                }
            },
            "check_url": {
                "keywords": ["检查网址", "网址状态", "url status", "网站是否可达", "检查网站", "网站检查", "检查链接", "链接可用", "检查", "是否可用"],
                "tool": "check_url_status",
                "priority": 6,
                "param_extractors": {
                    "url": self._extract_url
                }
            },
            "fetch_json": {
                "keywords": ["获取json", "api数据", "json数据", "fetch json", "api"],
                "tool": "fetch_json",
                "param_extractors": {
                    "url": self._extract_url
                }
            }
        }

        # 尝试导入 NLP 处理器
        self._nlp_processor = None
        try:
            from .nlp_processor import get_nlp_processor
            self._nlp_processor = get_nlp_processor()
        except ImportError:
            pass

    def _extract_code_block(self, text: str) -> Optional[str]:
        """提取代码块"""
        patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'代码[:：]\s*(.+)',
            r'运行[:：]\s*(.+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        math_patterns = [
            r'算\s*([\d\s\+\-\*\/\(\)]+)',
            r'计算\s*([\d\s\+\-\*\/\(\)]+)',
            r'(\d+\s*[\+\-\*/\(\)\s]+\d+)',
        ]
        for pattern in math_patterns:
            match = re.search(pattern, text)
            if match:
                expr = match.group(1).strip()
                if any(c.isdigit() for c in expr):
                    return f"print({expr})"
        
        return None

    def _extract_file_path(self, text: str) -> Optional[str]:
        """提取文件路径"""
        patterns = [
            r'["\']([^"\']+\.(csv|xlsx?|txt|json|xml))["\']',
            r'(?:file|path|文件)[:：]\s*([^\s]+)',
            r'(?:在|打开)\s+([^\s]+\.(csv|xlsx?|txt|json|xml))',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_directory(self, text: str) -> Optional[str]:
        """提取目录路径"""
        patterns = [
            r'(?:目录|文件夹|directory)[:：]\s*([^\s]+)',
            r'(?:在|从)\s+([^\s]+)\s+(?:目录|文件夹)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return "."

    def _extract_old_path(self, text: str) -> Optional[str]:
        """提取旧路径"""
        patterns = [
            r'(?:从|原|旧)[:：]\s*([^\s]+)',
            r'([^\s]+\.(csv|xlsx?|txt|json|xml))\s*(?:改|rename)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    def _extract_new_name(self, text: str) -> Optional[str]:
        """提取新名称"""
        patterns = [
            r'(?:改|新)[:：]\s*([^\s]+)',
            r'(?:to|->)\s*([^\s]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    def _extract_json_data(self, text: str) -> Optional[List[Dict]]:
        """提取JSON数据列表"""
        patterns = [
            r'(\[.*?\])',
            r'data\s*[:：]\s*(\[.*?\])',
            r'(\{.*?"name".*?"age".*?\})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    data_str = match.group(1)
                    if data_str.startswith('[') or data_str.startswith('{'):
                        parsed = json.loads(data_str)
                        if isinstance(parsed, list):
                            return parsed
                        elif isinstance(parsed, dict):
                            return [parsed]
                except json.JSONDecodeError:
                    continue
        return None

    def _extract_export_file_path(self, text: str) -> Optional[str]:
        """提取导出文件路径"""
        patterns = [
            (r'文件路径\s*[:：]\s*([^\s,\]]+)', 1),
            (r'输出\s*[:：]\s*([^\s,\]]+)', 1),
            (r'到\s+([^\s]+\.csv)', 1),
            (r'(test_output\.csv|output\.csv)', 1),
            (r'["\']([^"\']+\.csv)["\']', 1),
        ]
        for pattern, group_idx in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(group_idx).strip()
        return None

    def _extract_columns(self, text: str) -> Optional[List[str]]:
        """提取列名列表"""
        patterns = [
            r'columns?\s*[:：]\s*\[(.*?)\]',
            r'列\s*[:：]\s*\[(.*?)\]',
            r'字段\s*[:：]\s*\[(.*?)\]',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                columns_str = match.group(1)
                columns = [c.strip().strip('"\'') for c in columns_str.split(',')]
                if columns:
                    return columns
        return None

    def _extract_format(self, text: str) -> Optional[str]:
        """提取格式"""
        patterns = [
            r'(?:转|转换|to)\s*(json|csv|txt|xml)',
            r'格式[:：]\s*(json|csv|txt|xml)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        return None

    def _extract_time(self, text: str) -> Optional[str]:
        """提取时间"""
        patterns = [
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(今天|昨天|明天|now|today)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def _extract_start_time(self, text: str) -> Optional[str]:
        """提取开始时间"""
        patterns = [
            (r'从\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})', 1),
            (r'从\s*(\d{4}年\d{1,2}月\d{1,2}日)', 1),
            (r'开始[:：]\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})', 1),
            (r'开始[:：]\s*(\d{4}年\d{1,2}月\d{1,2}日)', 1),
        ]
        for pattern, group_idx in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(group_idx)
        return None

    def _extract_end_time(self, text: str) -> Optional[str]:
        """提取结束时间"""
        patterns = [
            (r'到\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})', 1),
            (r'到\s*(\d{4}年\d{1,2}月\d{1,2}日)', 1),
            (r'结束[:：]\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})', 1),
            (r'结束[:：]\s*(\d{4}年\d{1,2}月\d{1,2}日)', 1),
        ]
        for pattern, group_idx in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(group_idx)
        return None

    def _extract_time_value(self, text: str) -> Optional[int]:
        """提取时间值"""
        patterns = [
            r'(\d+)\s*(?:天|日|周|月|年|hour|min|sec|second|minute|day|week|month|year)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        return 1

    def _extract_time_unit(self, text: str) -> Optional[str]:
        """提取时间单位"""
        patterns = [
            (r'(\d+)\s*天', 'days'),
            (r'(\d+)\s*日', 'days'),
            (r'(\d+)\s*周', 'weeks'),
            (r'(\d+)\s*月', 'months'),
            (r'(\d+)\s*年', 'years'),
            (r'(\d+)\s*hour', 'hours'),
            (r'(\d+)\s*min', 'minutes'),
            (r'(\d+)\s*second', 'seconds'),
        ]
        for pattern, unit in patterns:
            if re.search(pattern, text):
                return unit
        return 'days'

    def _extract_birth_date(self, text: str) -> Optional[str]:
        """提取出生日期"""
        patterns = [
            (r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', 1),
            (r'(\d{4}年\d{1,2}月\d{1,2}日)', 1),
            (r'出生[:：]\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})', 1),
            (r'出生[:：]\s*(\d{4}年\d{1,2}月\d{1,2}日)', 1),
            (r'(出生于?\s*\d{4}[-/]\d{1,2}[-/]\d{1,2})', 1),
            (r'(出生于?\s*\d{4}年\d{1,2}月\d{1,2}日)', 1),
        ]
        for pattern, group_idx in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(group_idx)
        return None

    def _extract_url(self, text: str) -> Optional[str]:
        """提取URL"""
        pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(pattern, text)
        if match:
            return match.group(0)
        return None

    def _extract_post_data(self, text: str) -> Optional[Dict]:
        """提取POST数据"""
        patterns = [
            r'data[:：]\s*(\{[^}]+\})',
            r'(\{[^}]+\})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
        return {}

    def analyze(self, user_message: str) -> IntentAnalysis:
        """
        分析用户意图

        结合 NLP 预处理和关键词匹配来识别用户意图：
        1. 使用 jieba 进行中文分词
        2. 移除 JSON 数据避免干扰
        3. 按优先级匹配关键词
        4. 计算置信度

        Args:
            user_message: 用户输入的消息

        Returns:
            IntentAnalysis: 意图分析结果
        """
        import re

        # Step 1: 使用 NLP 预处理提取关键信息
        nlp_info = {}
        if self._nlp_processor:
            try:
                nlp_info = self._nlp_processor.preprocess_for_intent(user_message)
            except Exception:
                pass

        # Step 2: 移除 JSON 数据，避免数据中的关键词干扰匹配
        json_pattern = r'(\{[^{}]*\}|\[[^\[\]]*\])'
        message_without_json = re.sub(json_pattern, ' ', user_message, flags=re.DOTALL)
        message_without_json = ' '.join(message_without_json.split())
        message_lower = message_without_json.lower()

        # Step 3: 预处理分词后的文本（如果 NLP 可用）
        tokens_lower = []
        if nlp_info and 'tokens' in nlp_info:
            tokens_lower = [t.lower() for t in nlp_info['tokens']]

        # Step 4: 按优先级排序意图模式
        sorted_patterns = sorted(
            self.intent_patterns.items(),
            key=lambda x: x[1].get("priority", 0),
            reverse=True
        )

        # Step 5: 匹配意图
        best_match = None
        best_score = 0.0
        best_keyword = None

        for intent_name, intent_config in sorted_patterns:
            matched_keywords = []
            match_score = 0.0

            # 精确关键词匹配
            for keyword in intent_config["keywords"]:
                keyword_lower = keyword.lower()
                if keyword_lower in message_lower:
                    matched_keywords.append(keyword)
                    # 基础分数 + 关键词长度加权（越长越精确）
                    match_score += 1.0 + (len(keyword) * 0.1)

                # 如果 NLP 可用，也检查分词后的结果
                if tokens_lower and keyword_lower in tokens_lower:
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)
                    match_score += 1.5  # 分词匹配权重更高

            if matched_keywords and match_score > best_score:
                best_match = intent_name
                best_score = match_score
                best_keyword = matched_keywords[0]

        # Step 6: 如果找到匹配，提取参数并计算置信度
        if best_match and best_keyword:
            intent_config = self.intent_patterns[best_match]
            tool_name = intent_config["tool"]
            parameters = {}

            # 提取参数
            for param_name, extractor in intent_config["param_extractors"].items():
                try:
                    extracted = extractor(user_message)
                    if extracted:
                        parameters[param_name] = extracted
                except Exception:
                    pass

            # 计算置信度：基于匹配分数和参数提取率
            param_count = len(intent_config["param_extractors"])
            extracted_count = len(parameters)
            param_score = (extracted_count / param_count) * 0.3 if param_count > 0 else 0.3

            # 最终置信度 = 匹配分数归一化 + 参数提取分
            confidence = min(0.95, 0.5 + param_score + (best_score * 0.05))

            return IntentAnalysis(
                needs_tool=True,
                intent_type=best_match,
                confidence=confidence,
                suggested_tool=tool_name,
                parameters=parameters,
                reasoning=f"检测到关键词: {best_keyword}，提取到 {extracted_count}/{param_count} 个参数"
            )

        return IntentAnalysis(
            needs_tool=False,
            intent_type="general_conversation",
            confidence=0.0
        )


class ToolAgent:
    """工具Agent - 负责工具调用决策和执行"""

    def __init__(self, registry: registry = None):
        self.registry = registry or get_registry()
        self.rule_intent_recognizer = IntentRecognizer()
        self.llm_intent_recognizer = LLMIntentRecognizer()
        self.tool_descriptions = self.registry.get_tool_descriptions()

    def analyze_intent(self, user_message: str, use_llm: bool = True) -> IntentAnalysis:
        """分析用户意图

        Args:
            user_message: 用户输入的消息
            use_llm: 是否优先使用LLM进行意图分析，默认True

        Returns:
            IntentAnalysis: 意图分析结果
        """
        if use_llm:
            llm_result = self.llm_intent_recognizer.analyze(user_message, self.registry)
            if llm_result.suggested_tool:
                return llm_result

        rule_result = self.rule_intent_recognizer.analyze(user_message)
        if rule_result.suggested_tool:
            return rule_result

        return IntentAnalysis(
            needs_tool=False,
            intent_type="general_conversation",
            confidence=0.0,
            reasoning="未识别到工具调用需求"
        )

    def prepare_tool_call(self, intent: IntentAnalysis) -> Optional[ToolCallRequest]:
        """准备工具调用"""
        if not intent.needs_tool or not intent.suggested_tool:
            return None

        tool_func = self.registry.get_tool(intent.suggested_tool)
        if not tool_func:
            return None

        metadata = self.registry.get_metadata(intent.suggested_tool)
        danger_level = metadata.danger_level if metadata else "safe"

        needs_confirmation = danger_level in ["medium", "high"]

        return ToolCallRequest(
            tool_name=intent.suggested_tool,
            parameters=intent.parameters,
            confidence=intent.confidence,
            reasoning=intent.reasoning,
            user_confirmation_needed=needs_confirmation
        )

    def execute_tool(self, tool_call: ToolCallRequest) -> ToolCallResult:
        """执行工具调用"""
        import time
        start_time = time.time()

        tool_func = self.registry.get_tool(tool_call.tool_name)
        if not tool_func:
            return ToolCallResult(
                tool_name=tool_call.tool_name,
                success=False,
                error=f"工具不存在: {tool_call.tool_name}"
            )

        try:
            result = tool_func(**tool_call.parameters)
            execution_time = time.time() - start_time

            if isinstance(result, dict):
                success = result.get("success", True)
            else:
                success = True

            return ToolCallResult(
                tool_name=tool_call.tool_name,
                success=success,
                result=result,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ToolCallResult(
                tool_name=tool_call.tool_name,
                success=False,
                error=str(e),
                execution_time=execution_time
            )

    def format_tool_result(self, result: ToolCallResult) -> str:
        """格式化工具结果"""
        if result.success:
            output = f"✅ 工具 [{result.tool_name}] 执行成功\n"
            output += f"⏱️ 耗时: {result.execution_time:.3f}s\n\n"

            if isinstance(result.result, dict):
                if "message" in result.result:
                    output += f"📝 结果: {result.result['message']}\n"
                if "output" in result.result:
                    output += f"📤 输出:\n{result.result['output']}\n"
                if "file_info" in result.result:
                    output += f"📄 文件信息: {json.dumps(result.result['file_info'], ensure_ascii=False, indent=2)}\n"
                if "preview" in result.result:
                    preview_data = result.result['preview']
                    if isinstance(preview_data, list) and len(preview_data) > 0:
                        output += f"📊 数据预览 (前{len(preview_data)}行):\n"
                        for i, row in enumerate(preview_data[:5], 1):
                            output += f"  {i}. {row}\n"
                if "difference" in result.result:
                    diff = result.result['difference']
                    output += f"⏳ 时间差: {diff['value']} {diff['unit']}\n"
                if "result" in result.result:
                    output += f"🕐 计算结果: {result.result['result']}\n"
                if "data" in result.result:
                    output += f"📦 数据: {json.dumps(result.result['data'], ensure_ascii=False)[:500]}...\n"
            else:
                output += f"📤 结果: {result.result}\n"

            return output
        else:
            return f"❌ 工具 [{result.tool_name}] 执行失败\n💥 错误: {result.error}"

    def generate_confirmation_message(self, tool_call: ToolCallRequest) -> str:
        """生成确认消息"""
        tool_meta = self.registry.get_metadata(tool_call.tool_name)
        tool_desc = tool_meta.description if tool_meta else ""

        message = f"🔧 即将执行工具: [{tool_call.tool_name}]\n"
        message += f"📋 说明: {tool_desc}\n"
        message += f"⚠️ 此操作需要确认\n\n"
        message += f"📥 参数:\n"

        for key, value in tool_call.parameters.items():
            message += f"  - {key}: {value}\n"

        message += f"\n是否继续执行？回复\"确认\"或\"取消\""

        return message

    def get_tools_summary(self) -> str:
        """获取工具摘要"""
        return self.registry.generate_tools_prompt()


def create_agent() -> ToolAgent:
    """创建Agent实例"""
    return ToolAgent()
