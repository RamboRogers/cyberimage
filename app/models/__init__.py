"""
Models initialization for CyberImage
"""
from app.utils.config import get_available_models

# Get available models from configuration
AVAILABLE_MODELS = get_available_models()

# Set the default model (use the first model or flux-1 if available)
DEFAULT_MODEL = next(
    (model_id for model_id in AVAILABLE_MODELS if model_id == "flux-1"),
    next(iter(AVAILABLE_MODELS.keys())) if AVAILABLE_MODELS else None
)

# Make models accessible from this module
__all__ = ["AVAILABLE_MODELS", "DEFAULT_MODEL"]