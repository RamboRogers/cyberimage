"""
Models initialization for CyberImage
"""
from app.utils.config import get_available_models

# Get available models from configuration
AVAILABLE_MODELS = get_available_models()

# Make models accessible from this module
__all__ = ["AVAILABLE_MODELS"]