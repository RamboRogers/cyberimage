"""
Configuration utilities for CyberImage
"""
import os
import logging
import json
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def parse_model_config() -> Dict[str, Dict[str, Any]]:
    """
    Parse model configuration from environment variables.

    Format: MODEL_<N>=<name>;<repo>;<description>;<source>;<requires_auth>[;<options_json>]
    Source options: huggingface (local), huggingface_api, fal_api
    Example Local: MODEL_1=flux-1;black-forest-labs/FLUX.1-dev;FLUX base model;huggingface;true
    Example HF API: MODEL_10=llava-hf/llava-1.5-7b-hf;llava-hf/llava-1.5-7b-hf;LLaVA 1.5 7B HF;huggingface_api;true;{\"provider\": \"huggingface-inference-api\", \"type\": \"vqa\"}
    Example Fal API: MODEL_11=ltx-video-i2v-api;Lightricks/LTX-Video;LTX Image-to-Video;fal_api;true;{\"provider\": \"fal-ai\", \"type\": \"i2v\", \"fal_function_id\": \"fal-ai/ltx-video-13b-dev/image-to-video\"}

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
        "sdxl": [
            "model_index.json",
            "vae/diffusion_pytorch_model.safetensors"
        ],
        "animagine-xl": [
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
            "text_encoder_2/model.safetensors"
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

                # Updated check for minimum parts (5 required, 6th optional)
                if len(parts) < 5:
                    logger.warning(f"Invalid model configuration format for {key}: {value} (needs at least 5 parts)")
                    continue

                name, repo, description, source, requires_auth = parts[:5]
                options_json = parts[5] if len(parts) > 5 else None
                name = name.strip()

                # Parse the optional JSON configuration
                step_config = {} 
                if options_json:
                    options_json_stripped = options_json.strip('"\' ') # Clean up outer quotes/spaces
                    options_json_cleaned = options_json_stripped.replace('\\"', '"')
                    try:
                        parsed_options = json.loads(options_json_cleaned)
                        if isinstance(parsed_options, dict):
                            step_config = parsed_options
                        else:
                           logger.warning(f"Optional config for {key} is not a JSON object: {options_json_cleaned}")
                    except json.JSONDecodeError as json_err:
                        logger.warning(f"Invalid JSON in optional config for {key}: {options_json_cleaned} - Error: {json_err}")
                # else:
                    # print(f"DEBUG: No options_json found for {key}")

                # Further strip any quotes from individual parts
                name = name.strip('"\'')
                repo = repo.strip('"\'')
                description = description.strip('"\'')
                source_val = source.strip('"\'').lower() # Renamed to avoid conflict with 'source' module, and pre-process
                requires_auth = requires_auth.strip('"\'')

                # Determine if model is enabled based on its type
                # Models from API sources (huggingface_api, fal_api) are enabled by default unless explicitly disabled in options
                if source_val in ["huggingface_api", "fal_api"]:
                    download_enabled = step_config.get('download_enabled', True)
                    api_source_type = source_val.replace("_api", "").capitalize()
                    logger.info(f"{api_source_type} API model {name} is {'enabled' if download_enabled else 'disabled'} for use")
                else: # Local models
                    # For local models, check DOWNLOAD_MODEL_N environment variable
                    download_key = f"DOWNLOAD_MODEL_{model_num}"
                    download_value = os.environ.get(download_key, "true")
                    # Strip quotes from download value too
                    if (download_value.startswith('"') and download_value.endswith('"')) or (download_value.startswith("'") and download_value.endswith("'")):
                        download_value = download_value[1:-1]
                    download_enabled = download_value.lower() == "true"

                if not download_enabled:
                    logger.info(f"Model {name} is defined but disabled for use")

                # Determine model type from name (heuristic, can be overridden by JSON)
                heuristic_model_type = "generic" # Start with generic
                if "flux" in name.lower():
                    heuristic_model_type = "flux"
                elif "sd-3" in name.lower():
                    heuristic_model_type = "sd3"
                elif "animagine" in name.lower():
                    heuristic_model_type = "sdxl"  # Animagine XL uses SDXL architecture
                elif "xl" in name.lower() or "sdxl" in name.lower():
                    heuristic_model_type = "sdxl"

                # Get appropriate file list based on model name or heuristic type
                files = []
                if source_val != "huggingface_api":  # Only need files for local models
                    if name in common_file_lists:
                        files = common_file_lists[name]
                    elif heuristic_model_type in common_file_lists:
                        files = common_file_lists[heuristic_model_type]

                # If source is an API source (huggingface_api, fal_api), files are not applicable locally
                if source_val in ["huggingface_api", "fal_api"]:
                    files = [] # API models don't have local files for completeness check
                    logger.debug(f"Model {name} is from {source_val}, setting files to empty list.")

                # Validate provider for API sources
                if source_val in ["huggingface_api", "fal_api"]:
                    if not isinstance(step_config, dict) or "provider" not in step_config:
                        logger.error(f"Model {name} with source '{source_val}' is missing 'provider' in its JSON options. Skipping this model.")
                        continue
                    if not isinstance(step_config.get("provider"), str) or not step_config.get("provider").strip():
                        logger.error(f"Model {name} with source '{source_val}' has an invalid or empty 'provider' in its JSON options: '{step_config.get('provider')}'. Skipping this model.")
                        continue
                    logger.info(f"Configured {source_val} model {name} with provider: {step_config['provider']}")

                # Determine final model type: Explicit JSON type > Heuristic > Default ('image')
                final_model_type = 'image' # Default to image
                if heuristic_model_type != 'generic': # If heuristic found something specific
                    final_model_type = heuristic_model_type

                # --- Override model_type if specified in step_config --- #
                if isinstance(step_config, dict) and 'type' in step_config:
                    explicit_type = step_config['type']
                    if isinstance(explicit_type, str) and explicit_type:
                        logger.debug(f"Overriding model type with explicit type '{explicit_type}' from JSON options for {name}")
                        final_model_type = explicit_type.lower() # Use explicit type, lowercased
                    else:
                        logger.warning(f"'type' found in JSON options for {name}, but it's not a valid string: {explicit_type}")
                # --- End Override --- #

                # --- Standardize non-video types to 'image' --- #
                if final_model_type not in ['t2v', 'i2v']:
                    if final_model_type != 'image': # Log if we are changing it
                        logger.debug(f"Standardizing model type '{final_model_type}' to 'image' for {name}")
                    final_model_type = 'image'
                # --- End Standardization --- #

                models_config[name] = {
                    "repo": repo.strip(),
                    "description": description.strip(),
                    "source": source_val, # Use the processed source_val
                    "requires_auth": requires_auth.lower() == "true",
                    "download_enabled": download_enabled, # Represents if the model (local or API) is active
                    "type": final_model_type, # Use the determined final type
                    "files": files, # Will be empty for API sources like huggingface_api or fal_api
                    "step_config": step_config # Add parsed step config (contains provider for API models)
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
    parsed_models = parse_model_config()
    ui_models = {}

    # Add display information and standardize format for UI
    for name, config in parsed_models.items():
        # Start with a copy of the full original config to preserve all fields
        model_entry = config.copy()
        
        # Set/Override UI-specific fields
        model_entry["id"] = config.get("repo", name) # Use repo as ID, fallback to name
        model_entry["name"] = config.get("display_name", name) # Use display_name, fallback to name (original key)
        
        # Ensure 'source' is definitely there (it should be from parse_model_config)
        if 'source' not in model_entry:
            logger.warning(f"Source field was missing from parsed_config for {name} in get_available_models. This is unexpected.")
            model_entry['source'] = config.get('source', 'unknown') # Should already be there

        ui_models[name] = model_entry

    return ui_models

# --- Rate Limit Configuration ---
ENABLE_RATE_LIMIT = False  # Set to False to disable IP-based hourly rate limiting
RATE_LIMIT_HOURLY = 10     # Default: Max 10 requests per hour per IP (only applies if ENABLE_RATE_LIMIT is True)
# --- End Rate Limit Configuration ---