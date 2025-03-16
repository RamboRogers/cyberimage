"""
Configuration utilities for CyberImage
"""
import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def parse_model_config() -> Dict[str, Dict[str, Any]]:
    """
    Parse model configuration from environment variables.

    Format: MODEL_<N>=<name>;<repo>;<description>;<source>;<requires_auth>
    Example: MODEL_1=flux-1;black-forest-labs/FLUX.1-dev;FLUX base model;huggingface;true

    Returns:
        Dict mapping model names to their configurations
    """
    models_config = {}
    enabled_models = set()

    # Common file lists for model completeness checks
    common_file_lists = {
        "flux": [
            "model_index.json",
            "ae.safetensors"
            # Removed flux1-dev.safetensors to be more forgiving
        ],
        "flux-schnell": [
            "model_index.json",
            "ae.safetensors"
            # No specific safetensors file check since it may vary
        ],
        "sd3": [
            "model_index.json",
            "sd3.5_large.safetensors"
        ],
        "flux-abliterated": [
            "model_index.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors"
        ]
    }

    # Find all model definitions in environment variables
    for key, value in os.environ.items():
        # Skip MODEL_FOLDER which is a path setting, not a model definition
        if key == 'MODEL_FOLDER':
            continue

        if key.startswith('MODEL_') and value and '_' not in key[6:]:
            try:
                model_num = key[6:]  # Extract number from MODEL_N

                # Strip quotes from the value
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]

                parts = value.split(';')

                if len(parts) < 5:
                    logger.warning(f"Invalid model configuration format for {key}: {value}")
                    continue

                name, repo, description, source, requires_auth = parts[:5]
                name = name.strip()

                # Further strip any quotes from individual parts
                name = name.strip('"\'')
                repo = repo.strip('"\'')
                description = description.strip('"\'')
                source = source.strip('"\'')
                requires_auth = requires_auth.strip('"\'')

                # Check if this model is enabled for download
                download_key = f"DOWNLOAD_MODEL_{model_num}"
                download_value = os.environ.get(download_key, "true")
                # Strip quotes from download value too
                if (download_value.startswith('"') and download_value.endswith('"')) or (download_value.startswith("'") and download_value.endswith("'")):
                    download_value = download_value[1:-1]
                download_enabled = download_value.lower() == "true"

                if not download_enabled:
                    logger.info(f"Model {name} is defined but disabled for download")

                # Determine model type from name for automatic file lists
                model_type = "generic"
                if "flux" in name.lower():
                    model_type = "flux"
                elif "sd-3" in name.lower():
                    model_type = "sd3"

                # Get appropriate file list based on model name or type
                files = []
                if name in common_file_lists:
                    files = common_file_lists[name]
                elif model_type in common_file_lists:
                    files = common_file_lists[model_type]

                models_config[name] = {
                    "repo": repo.strip(),
                    "description": description.strip(),
                    "source": source.strip().lower(),
                    "requires_auth": requires_auth.strip().lower() == "true",
                    "download_enabled": download_enabled,
                    "type": model_type,
                    "files": files
                }

                # Store enabled models
                if download_enabled:
                    enabled_models.add(name)

                logger.debug(f"Loaded model configuration for {name}")

            except Exception as e:
                logger.warning(f"Error parsing model configuration {key}: {str(e)}")

    # If no models are defined in environment, use default configuration
    if not models_config:
        logger.warning("No models found in environment variables, using defaults")
        models_config = {
            "flux-1": {
                "repo": "black-forest-labs/FLUX.1-dev",
                "description": "FLUX base model",
                "requires_auth": True,
                "source": "huggingface",
                "download_enabled": True,
                "type": "flux",
                "files": common_file_lists["flux"]
            },
            "sd-3.5": {
                "repo": "stabilityai/stable-diffusion-3.5-large",
                "description": "Stable Diffusion 3.5",
                "requires_auth": True,
                "source": "huggingface",
                "download_enabled": True,
                "type": "sd3",
                "files": common_file_lists["sd3"]
            },
            "flux-abliterated": {
                "repo": "aoxo/flux.1dev-abliteratedv2",
                "description": "FLUX Abliterated variant",
                "requires_auth": True,
                "source": "huggingface",
                "download_enabled": True,
                "type": "flux",
                "files": common_file_lists["flux-abliterated"]
            }
        }
        enabled_models = set(models_config.keys())

    # Log summary of configured models
    logger.info(f"Loaded {len(models_config)} model configurations")
    logger.info(f"Models enabled for download: {', '.join(enabled_models)}")

    return models_config

def get_downloadable_models() -> Dict[str, Dict[str, Any]]:
    """
    Get only the models that are enabled for download

    Returns:
        Dict mapping model names to their configurations for enabled models
    """
    all_models = parse_model_config()
    return {name: config for name, config in all_models.items()
            if config.get("download_enabled", True)}

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get all available models for the UI (includes display metadata)

    Returns:
        Dict mapping model names to their UI configurations
    """
    models = parse_model_config()

    # Add display information and standardize format for UI
    for name, config in models.items():
        models[name]["id"] = name
        models[name]["name"] = config.get("display_name", name)

    return models