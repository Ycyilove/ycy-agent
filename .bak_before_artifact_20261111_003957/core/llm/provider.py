"""
惰性加载设计：
    - 顶层不 import anthropic / langchain_openai / langchain_huggingface /
      langchain_community / langchain_core / langchain_text_splitters
    - 所有重模块在 __init__ 或首次调用时加载
    - 使用 lazy_loader 与 app.py 的模块注册保持一致
"""

import json
from typing import Dict, List, Optional
from core import ports


def _cfg():
    """延迟读配置，避免 core -> backend 模块级依赖。"""
    cfg = ports.get_config()
    if cfg is None:
        raise RuntimeError(
            "core.ports.config 未注入，bootstrap/startup 未执行"
        )
    return cfg
from .local_model import get_local_model_service, ModelSource

import logging
logger = logging.getLogger(__name__)

class ContextLengthExceeded(RuntimeError):
    """prompt 超出模型 context 上限。

    由 provider 层抛，上层（_decide / _fill_params）catch 后降级重试。
    """


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

    def __init__(self, model: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 2048):
        if model is None:
            model = _cfg().DEFAULT_MODEL
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local = _cfg().MODEL_SOURCE == ModelSource.LOCAL
        self.llm = None          # 延迟到首次使用
        self.local_service = None

        if self.use_local:
            self.local_service = get_local_model_service(_cfg().LOCAL_MODEL_CONFIG)

    def _ensure_llm(self):
        """确保 self.llm 已初始化。

        本方法会阻塞调用线程（首次约 30 秒）。
        必须在线程池 worker 里调用。
        """
        if self.llm is not None:
            return
        # 用 lazy_loader 统一管理（与 app.py 共享同一个注册表）
        try:
            from infra.lazy_loader import ensure_module_loaded_sync
            ensure_module_loaded_sync("langchain_openai")
        except Exception:
            # lazy_loader 未注册/不可用时，降级为直接 import
            # 注：实际抛的是 KeyError，不是 RuntimeError
            pass
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=_cfg().YUNDU_API_KEY,
            base_url=_cfg().YUNDU_BASE_URL,
            max_retries=2
        )

    def generate(self, messages, mode='quick', timeout=None) -> str:
        processed_messages = self._add_mode_prompt(messages, mode)
        if self.use_local:
            if not self.local_service or not self.local_service.is_loaded:
                raise RuntimeError('本地模型未加载，请确保 MODEL_SOURCE 设置正确')
            return self.local_service.generate(processed_messages)

        self._ensure_llm()
        try:
            return self.llm.invoke(
                self._convert_to_langchain(processed_messages),
                config=self._langchain_config(timeout=timeout),
            ).content
        except Exception as e:
            err = str(e).lower()
            if (
                "context_length" in err
                or "maximum context" in err
                or "too many tokens" in err
                or "reduce the length" in err
                or "prompt is too long" in err
            ):
                raise ContextLengthExceeded(
                    f"prompt 超模型 context 上限：{str(e)[:200]}"
                ) from e
            raise

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        mode: str = 'quick',
        timeout: float | None = None,
    ):
        processed_messages = self._add_mode_prompt(messages, mode)
        if self.use_local:
            if not self.local_service or not self.local_service.is_loaded:
                yield f"data: {json.dumps({'text': '本地模型未加载', 'error': True}, ensure_ascii=False)}\n\n"
                yield 'data: [DONE]\n\n'
                return
            yield from self.local_service.stream_generate(processed_messages)
            return

        self._ensure_llm()
        try:
            for chunk in self.llm.stream(
                self._convert_to_langchain(processed_messages),
                config=self._langchain_config(timeout=timeout),
            ):
                if chunk.content:
                    yield f"data: {json.dumps({'text': chunk.content}, ensure_ascii=False)}\n\n"
        except Exception as error:
            err = str(error).lower()
            if (
                "context_length" in err
                or "maximum context" in err
                or "too many tokens" in err
                or "reduce the length" in err
                or "prompt is too long" in err
            ):
                yield (
                    f"data: {json.dumps({'text': 'prompt 超模型 context 上限，请缩短任务或减少历史', 'error': True}, ensure_ascii=False)}\n\n"
                )
            else:
                yield f"data: {json.dumps({'text': f'生成响应时发生错误: {error}', 'error': True}, ensure_ascii=False)}\n\n"
        # 不在 finally 里 yield（close() 时抛 RuntimeError）
        yield 'data: [DONE]\n\n'

    def _add_mode_prompt(self, messages: List[Dict[str, str]], mode: str = 'quick'):
        system_prompt = self.QUICK_RESPONSE_PROMPT if mode == 'quick' else self.DEEP_THINKING_PROMPT
        return [{'role': 'system', 'content': system_prompt}, *messages]

    @staticmethod
    def _langchain_config(timeout: float | None = None):
        """构造 langchain 调用配置。

        - callbacks：从 observability 拿 per-turn handler（未启用时忽略）
        - timeout：透传给底层 OpenAI 客户端；超时后 httpx 会中断连接，
          线程池里的调用会真正返回，而不是一直挂着。
        """
        cfg: dict = {}

        try:
            from infra.observability import get_callback_handler
            handler = get_callback_handler()
            if handler is not None:
                cfg["callbacks"] = [handler]
        except ImportError:
            pass
        except Exception:
            pass

        if timeout is not None:
            cfg["timeout"] = float(timeout)

        return cfg or None

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
            from infra.lazy_loader import ensure_module_loaded_sync
            ensure_module_loaded_sync("anthropic")
        except Exception:
            pass
        import anthropic
        self.client = anthropic.Anthropic(
            base_url=_cfg().MODESCOPE_BASE_URL, api_key=_cfg().MODESCOPE_API_KEY,
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
