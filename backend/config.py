"""Runtime configuration shared by backend services."""

import os

from .services.local_model import LocalModelConfig, ModelSource

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '0')
os.environ.setdefault('HF_HUB_CACHE', os.path.join(PROJECT_ROOT, 'models'))

# ─────────────────────────────────────────────────────────────
# RAG / 检索 embedding 配置
# ─────────────────────────────────────────────────────────────

# RAG 知识库的 embedding 模型（中文优化）
# 换模型后必须清空 vector_store/ 重建索引（维度可能变化）
RAG_EMBEDDING_MODEL = os.getenv(
    'RAG_EMBEDDING_MODEL',
    'BAAI/bge-small-zh-v1.5',
)

# 工具筛选器的 embedding 模型（保持英文模型，工具名/描述主要是英文）
SHORTLIST_EMBEDDING_MODEL = os.getenv(
    'SHORTLIST_EMBEDDING_MODEL',
    'sentence-transformers/all-MiniLM-L6-v2',
)

# 检索混合策略
RAG_TOP_K_DEFAULT = int(os.getenv('RAG_TOP_K_DEFAULT', '15'))
RAG_BM25_ENABLED = os.getenv('RAG_BM25_ENABLED', 'true').lower() in ('1', 'true', 'yes')
RAG_QUERY_REWRITE_ENABLED = os.getenv('RAG_QUERY_REWRITE_ENABLED', 'true').lower() in ('1', 'true', 'yes')
RAG_RRF_K = int(os.getenv('RAG_RRF_K', '20'))

def _require_env(name: str, allow_empty: bool = False) -> str:
    """必须从 .env 读取的敏感配置。缺失时启动即报错，避免静默用占位值。"""
    value = os.getenv(name, '').strip()
    if not value and not allow_empty:
        raise RuntimeError(
            f"缺少必需环境变量 {name}，请在 .env 中配置。"
            f"（不要硬编码到代码里——一旦提交就是永久泄漏）"
        )
    return value


MODESCOPE_API_KEY = _require_env('MODESCOPE_API_KEY')
MODESCOPE_BASE_URL = 'https://api-inference.modelscope.cn/v1'
SILICONFLOW_API_KEY = _require_env('SILICONFLOW_API_KEY')
SILICONFLOW_BASE_URL = 'https://api.siliconflow.cn/v1'
YUNDU_API_KEY = _require_env('YUNDU_API_KEY')
YUNDU_BASE_URL = 'https://yundu.lol/v1'
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-5.6-luna')
MODEL_SOURCE = ModelSource.ONLINE

LOCAL_MODEL_CONFIG = LocalModelConfig(
    model_name='deepseek-ai/DeepSeek-V4.1-Flash',
    model_path=os.getenv('LOCAL_MODEL_PATH'),
    device=os.getenv('DEVICE', 'cpu'),
    max_tokens=2048,
    temperature=0.7,
    use_modelscope=True,
)

# ─────────────────────────────────────────────────────────────
# 三方向重构开关（默认全部关闭，需要时在 .env 里逐个打开）
# ─────────────────────────────────────────────────────────────

# ── Langfuse 可观测性 ──
# 开关：LANGFUSE_ENABLED=true 且 PUBLIC/SECRET KEY 非空时才启用
LANGFUSE_ENABLED = os.getenv('LANGFUSE_ENABLED', 'false').lower() in ('1', 'true', 'yes')
# Langfuse 是可选功能：未配置时 observability 会静默降级，所以允许为空
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY', '').strip()
LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY', '').strip()
LANGFUSE_HOST = os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')

# ── Mem0 结构化记忆 ──
MEM0_ENABLED = os.getenv('MEM0_ENABLED', 'true').lower() in ('1', 'true', 'yes')
MEM0_API_KEY = os.getenv('MEM0_API_KEY', '')
MEM0_CONFIG = os.getenv('MEM0_CONFIG', '')

# ── E2B 沙箱 ──
# 开关：SANDBOX_BACKEND=e2b 且 E2B_API_KEY 非空时才启用；否则走本地沙箱
SANDBOX_BACKEND = os.getenv('SANDBOX_BACKEND', 'local')
E2B_API_KEY = os.getenv('E2B_API_KEY', '')
E2B_TEMPLATE = os.getenv('E2B_TEMPLATE', 'base')   # 预留给 JSON 覆盖

ALLOWED_CORS_ORIGINS = [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:5174',
    'http://127.0.0.1:5174',
]

# query 改写 LLM 调用的超时（秒），超时后降级为原 query
RAG_QUERY_REWRITE_TIMEOUT = int(os.getenv("RAG_QUERY_REWRITE_TIMEOUT", "5"))

# ─────────────────────────────────────────────────────────────
# Contextual Retrieval（情境化摘要）
# ─────────────────────────────────────────────────────────────

# 开关：入库时为每个 chunk 生成情境化摘要，提升检索召回
RAG_CONTEXTUAL_ENABLED = os.getenv(
    'RAG_CONTEXTUAL_ENABLED', 'false'
).lower() in ('1', 'true', 'yes')

# 生成摘要的并发数（SiliconFlow 有速率限制，建议 5-10）
RAG_CONTEXTUAL_CONCURRENCY = int(os.getenv('RAG_CONTEXTUAL_CONCURRENCY', '5'))

# 单 chunk LLM 调用超时（秒）
RAG_CONTEXTUAL_TIMEOUT = float(os.getenv('RAG_CONTEXTUAL_TIMEOUT', '30'))

# 短于此字符数的 chunk 跳过摘要（本身已带上下文）
RAG_CONTEXTUAL_MIN_CHARS = int(os.getenv('RAG_CONTEXTUAL_MIN_CHARS', '80'))