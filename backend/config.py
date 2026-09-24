"""Runtime configuration shared by backend services."""

import os

from .services.local_model import LocalModelConfig, ModelSource


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
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
MODEL_SOURCE = ModelSource.ONLINE

LOCAL_MODEL_CONFIG = LocalModelConfig(
    model_name='deepseek-ai/DeepSeek-V4.1-Flash',
    model_path=os.getenv('LOCAL_MODEL_PATH'),
    device=os.getenv('DEVICE', 'cpu'),
    max_tokens=2048,
    temperature=0.7,
    use_modelscope=True,
)

ALLOWED_CORS_ORIGINS = [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:5174',
    'http://127.0.0.1:5174',
]
