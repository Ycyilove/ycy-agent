"""Runtime configuration shared by backend services."""

import os

from .services.local_model import LocalModelConfig, ModelSource

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '0')
os.environ.setdefault('HF_HUB_CACHE', os.path.join(PROJECT_ROOT, 'models'))

MODESCOPE_API_KEY = os.getenv(
    'MODESCOPE_API_KEY',
    'ms-e0031b6d-4579-45d0-8e43-cd411cb9de1e',
)
MODESCOPE_BASE_URL = 'https://api-inference.modelscope.cn/v1'
SILICONFLOW_API_KEY = os.getenv(
    'SILICONFLOW_API_KEY',
    'sk-fkzisgswcbaplwndxwlrjnqayfjcpueizuhnwxcizfdgxglu',
)
SILICONFLOW_BASE_URL = 'https://api.siliconflow.cn/v1'
DEFAULT_MODEL = 'Qwen/Qwen3-8B'
# DEFAULT_MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'
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
LANGFUSE_ENABLED = os.getenv('LANGFUSE_ENABLED', 'true').lower() in ('1', 'true', 'yes')
LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-b6ce2022-518a-4d12-a9d0-49db68702d34')
LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY', 'sk-lf-d8d61132-ab0d-4a01-9441-e3eaa5946fff')
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
