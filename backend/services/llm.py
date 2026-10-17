"""Model and legacy text-RAG integrations.

惰性加载设计：
    - 顶层不 import anthropic / langchain_openai / langchain_huggingface /
      langchain_community / langchain_core / langchain_text_splitters
    - 所有重模块在 __init__ 或首次调用时加载
    - 使用 lazy_loader 与 app.py 的模块注册保持一致
"""

import json
from typing import Dict, List

from ..config import (
    DEFAULT_MODEL,
    LOCAL_MODEL_CONFIG,
    MODESCOPE_API_KEY,
    MODESCOPE_BASE_URL,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    MODEL_SOURCE,
)
from .local_model import get_local_model_service, ModelSource


class ModelScopeLLM:
    """OpenAI-compatible online model or local model adapter.

    关键设计：
        - __init__ 不加载 langchain_openai，只声明 use_local
        - 真正的加载发生在 generate / stream_generate 里
          （这些方法由 FastAPI 在 thread pool 里调用，或由 sync 路由直接执行）
        - _convert_to_langchain 里惰性 import langchain_core.messages
    """

    QUICK_RESPONSE_PROMPT = '你是一个高效的AI助手，请直接、简洁地回答用户问题，不要过度思考，快速给出答案。'
    DEEP_THINKING_PROMPT = '你是一个善于深度思考的AI助手，请仔细分析问题，一步步推理，先给出思考过程，再给出最终答案。请用"思考："开头来展示你的思考过程，然后用"答案："来给出最终结论。'

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.7, max_tokens: int = 2048):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local = MODEL_SOURCE == ModelSource.LOCAL
        self.llm = None          # 延迟到首次使用
        self.local_service = None

        if self.use_local:
            self.local_service = get_local_model_service(LOCAL_MODEL_CONFIG)

    def _ensure_llm(self):
        """确保 self.llm 已初始化。

        本方法会阻塞调用线程（首次约 30 秒）。
        必须在线程池 worker 里调用。
        """
        if self.llm is not None:
            return
        # 用 lazy_loader 统一管理（与 app.py 共享同一个注册表）
        try:
            from .lazy_loader import ensure_module_loaded_sync
            ensure_module_loaded_sync("langchain_openai")
        except (ImportError, RuntimeError):
            # 如果 lazy_loader 不可用，直接 import（降级）
            pass
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=SILICONFLOW_API_KEY,
            base_url=SILICONFLOW_BASE_URL,
        )

    def generate(self, messages: List[Dict[str, str]], mode: str = 'quick') -> str:
        processed_messages = self._add_mode_prompt(messages, mode)
        if self.use_local:
            if not self.local_service or not self.local_service.is_loaded:
                raise RuntimeError('本地模型未加载，请确保 MODEL_SOURCE 设置正确')
            return self.local_service.generate(processed_messages)

        self._ensure_llm()
        return self.llm.invoke(
            self._convert_to_langchain(processed_messages),
            config=self._langchain_config(),
        ).content

    def stream_generate(self, messages: List[Dict[str, str]], mode: str = 'quick'):
        processed_messages = self._add_mode_prompt(messages, mode)
        if self.use_local:
            if not self.local_service or not self.local_service.is_loaded:
                yield f"data: {json.dumps({'text': '本地模型未加载', 'error': True}, ensure_ascii=False)}\n\n"
                yield 'data: [DONE]\n\n'
                return
            yield from self.local_service.stream_generate(processed_messages)
            return

        self._ensure_llm()   # 在 thread pool 里加载，安全
        try:
            for chunk in self.llm.stream(
                self._convert_to_langchain(processed_messages),
                config=self._langchain_config(),
            ):
                if chunk.content:
                    yield f"data: {json.dumps({'text': chunk.content}, ensure_ascii=False)}\n\n"
        except Exception as error:
            yield f"data: {json.dumps({'text': f'生成响应时发生错误: {error}', 'error': True}, ensure_ascii=False)}\n\n"
        finally:
            yield 'data: [DONE]\n\n'

    def _add_mode_prompt(self, messages: List[Dict[str, str]], mode: str = 'quick'):
        system_prompt = self.QUICK_RESPONSE_PROMPT if mode == 'quick' else self.DEEP_THINKING_PROMPT
        return [{'role': 'system', 'content': system_prompt}, *messages]

    @staticmethod
    def _langchain_config():
        """从 observability 拿 per-turn callback handler。

        未启用 Langfuse 时返回 None，langchain 会忽略。
        """
        try:
            from .observability import get_callback_handler
        except ImportError:
            return None
        handler = get_callback_handler()
        if handler is None:
            return None
        return {"callbacks": [handler]}

    @staticmethod
    def _convert_to_langchain(messages: List[Dict[str, str]]):
        """惰性 import langchain_core.messages，避免顶层拉 langsmith。"""
        from langchain_core.messages import HumanMessage, SystemMessage
        converted = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if role == 'system':
                converted.append(SystemMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted


class MultimodalClient:
    """Anthropic-compatible multimodal adapter.

    关键设计：
        - __init__ 不加载 anthropic
        - 加载发生在 stream_analyze / analyze 里（thread pool 中）
    """

    def __init__(self):
        self.client = None   # 延迟到首次使用

    def _ensure_client(self):
        """确保 self.client 已初始化。必须在线程池里调用。"""
        if self.client is not None:
            return
        try:
            from .lazy_loader import ensure_module_loaded_sync
            ensure_module_loaded_sync("anthropic")
        except (ImportError, RuntimeError):
            pass
        import anthropic
        self.client = anthropic.Anthropic(
            base_url='https://api-inference.modelscope.cn',
            api_key=MODESCOPE_API_KEY,
        )

    def stream_analyze(self, messages: List[Dict], max_tokens: int = 4096):
        self._ensure_client()
        with self.client.messages.stream(
            model='deepseek-ai/DeepSeek-V4.1-Flash',
            messages=messages,
            max_tokens=max_tokens,
            extra_headers={'anthropic-beta': 'thinking-2025-01-21'},
        ) as stream:
            for event in stream:
                if getattr(event, 'type', None) == 'content_block_delta':
                    delta = getattr(event, 'delta', None)
                    if getattr(delta, 'type', None) == 'thinking_delta':
                        yield f"data: {json.dumps({'thinking': delta.thinking, 'type': 'thinking'})}\n\n"
                    elif getattr(delta, 'type', None) == 'text_delta':
                        yield f"data: {json.dumps({'text': delta.text, 'type': 'text'})}\n\n"
                    elif getattr(delta, 'text', None):
                        yield f"data: {json.dumps({'text': delta.text, 'type': 'text'})}\n\n"
                elif getattr(event, 'text', None):
                    yield f"data: {json.dumps({'text': event.text, 'type': 'text'})}\n\n"

    def analyze(self, messages: List[Dict], max_tokens: int = 2048) -> str:
        self._ensure_client()
        response = self.client.messages.create(
            model='deepseek-ai/DeepSeek-V4.1-Flash',
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.content[0].text


class RAGService:
    """Legacy in-memory Chroma RAG API kept for compatibility endpoints.

    关键设计：
        - __init__ 不加载任何 langchain 组件
        - 加载发生在 _ensure_components 里（thread pool 中）
    """

    def __init__(self, embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = None

    def _ensure_components(self):
        """确保所有组件已初始化。必须在线程池里调用。"""
        if self.embeddings is not None:
            return
        try:
            from .lazy_loader import ensure_module_loaded_sync
            ensure_module_loaded_sync("langchain_huggingface")
            ensure_module_loaded_sync("langchain_text_splitters")
        except (ImportError, RuntimeError):
            pass
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def add_texts(self, texts: List[str]):
        self._ensure_components()
        if not self.vector_store:
            try:
                from .lazy_loader import ensure_module_loaded_sync
                ensure_module_loaded_sync("langchain_community")
            except (ImportError, RuntimeError):
                pass
            from langchain_community.vectorstores import Chroma
            self.vector_store = Chroma.from_texts(texts=texts, embedding=self.embeddings)
        else:
            self.vector_store.add_texts(texts)
        return f'已添加 {len(texts)} 条文本'

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        self._ensure_components()
        if not self.vector_store:
            return ['知识库未初始化，请先加载文档']
        return [doc.page_content for doc in self.vector_store.similarity_search(query, k=top_k)]

    def rag_answer(self, query: str, llm: 'ModelScopeLLM' = None) -> str:
        self._ensure_components()
        llm = llm or ModelScopeLLM()
        context = '\n\n'.join(self.retrieve(query, top_k=3))
        prompt = f'''基于以下参考信息回答用户问题。如果参考信息不足以回答，请说明无法回答。

参考信息:
{context}

用户问题: {query}

回答:'''
        return llm.generate([{'role': 'user', 'content': prompt}])