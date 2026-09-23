"""Compatibility facade for the moved local model service."""

from backend.services.local_model import (
    LocalModelConfig,
    LocalModelService,
    ModelSource,
    download_model_from_modelscope,
    get_local_model_service,
)

__all__ = [
    "LocalModelConfig",
    "LocalModelService",
    "ModelSource",
    "download_model_from_modelscope",
    "get_local_model_service",
]
