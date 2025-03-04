"""
Model Definitions for CyberImage
"""
from typing import Dict, Any

AVAILABLE_MODELS = {
    "flux-1": {
        "id": "black-forest-labs/FLUX.1-dev",
        "description": "High-quality image generation model optimized for detailed outputs",
        "type": "flux"
    },
    "sd-3.5": {
        "id": "stabilityai/stable-diffusion-3.5-large",
        "description": "Latest Stable Diffusion model with improved quality and speed",
        "type": "sd"
    },
    "flux-abliterated": {
        "id": "aoxo/flux.1dev-abliteratedv2",
        "description": "Modified FLUX model with enhanced capabilities",
        "type": "flux"
    }
}

# Set the default model
DEFAULT_MODEL = "flux-1"