"""
Model management system for CyberImage
"""
import logging
import sys
import platform
import gc
import os
from typing import Dict, Optional, Union, Tuple, List
import torch
from diffusers import FluxPipeline, DiffusionPipeline, StableDiffusionPipeline, BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline, SanaSprintPipeline, WanPipeline
from app.models import AVAILABLE_MODELS
from tqdm import tqdm
from threading import Lock
from flask import current_app
import time
import threading
from pathlib import Path
from PIL import Image
import uuid
from huggingface_hub import InferenceClient
import base64 # Added
from io import BytesIO # Added

# Configure logging
logger = logging.getLogger(__name__)

# --- Add Video Generation Imports ---
import numpy as np
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
# --- End Video Generation Imports ---

# --- Add LTX-Video GGUF Imports --- #
from diffusers import LTXPipeline, LTXVideoTransformer3DModel, LTXImageToVideoPipeline
try:
    from diffusers.utils.import_utils import is_torch_available
except ImportError:
    from diffusers.utils import is_torch_available
# Use GGUFQuantizationConfig directly if available, otherwise define a placeholder or adjust logic
try:
    from diffusers import GGUFQuantizationConfig
except ImportError:
    # Define a placeholder or handle the absence of GGUFQuantizationConfig
    # This might require adjustments based on the diffusers version
    GGUFQuantizationConfig = None
    print("WARNING: diffusers.GGUFQuantizationConfig not found. GGUF loading might be affected.")
# --- End LTX-Video GGUF Imports --- #

class ModelLoadError(Exception):
    """Raised when a model fails to load"""
    pass

class ModelGenerationError(Exception):
    """Raised when image generation fails"""
    pass

def get_optimal_device():
    """Determine the best available compute device, requiring GPU acceleration"""
    if platform.system() == "Darwin":  # macOS
        if torch.backends.mps.is_available():
            logger.info("Using MPS (Metal Performance Shaders) on Apple Silicon")
            return "mps", torch.float16
        raise RuntimeError("MPS (Metal) acceleration is required but not available on this Mac")
    elif torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        logger.info(f"Using CUDA GPU: {device_name}")
        torch.cuda.empty_cache()
        gc.collect()
        return "cuda", torch.bfloat16

    raise RuntimeError("No GPU acceleration available. This application requires either CUDA (NVIDIA) or MPS (Apple Silicon)")

def progress_callback(state):
    """Callback for model download progress"""
    if state["status"] == "downloading":
        filename = state["filename"]
        progress = state["progress"]
        if progress is not None:
            sys.stdout.write(f"\r📥 Downloading {filename}: {progress:.2%}")
            sys.stdout.flush()
            if progress >= 1.0:
                sys.stdout.write("\n✅ Download complete!\n")
                sys.stdout.flush()

class ModelManager:
    """Manages AI models for image generation"""

    def __init__(self):
        # Set PyTorch CUDA memory allocation configuration to reduce fragmentation
        if torch.cuda.is_available():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            logger.info("Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

        # Initial memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.debug(f"Initial GPU memory: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

        self._loaded_models: Dict[str, Union[FluxPipeline, DiffusionPipeline]] = {}
        self._max_models = current_app.config["MODEL_CACHE_SIZE"]  # Get from config
        self._device, self._dtype = get_optimal_device()
        # Add global lock for model management
        self._model_lock = Lock()

        # Load HF_TOKEN for API-based models
        self._hf_token = current_app.config.get('HF_TOKEN')
        if not self._hf_token:
            logger.warning("HF_TOKEN not found in config. API-based models using default Hugging Face provider may not be available.")
        else:
            logger.info("HF_TOKEN loaded successfully.")

        # Load REPLICATE_API_KEY for Replicate provider
        self._replicate_api_key = current_app.config.get('REPLICATE_API_KEY')
        if not self._replicate_api_key:
            logger.warning("REPLICATE_API_KEY not found in config. Models using Replicate provider will not be available.")
        else:
            logger.info("REPLICATE_API_KEY loaded successfully.")

        # Load FAL_AI_API_KEY for fal-ai provider
        self._fal_ai_api_key = current_app.config.get('FAL_AI_API_KEY')
        if not self._fal_ai_api_key:
            logger.warning("FAL_AI_API_KEY not found in config. Models using fal-ai provider will not be available.")
        else:
            logger.info("FAL_AI_API_KEY loaded successfully.")

        # --- Add Cache Tracking State ---
        self._currently_loaded_model_key: Optional[str] = None
        self._last_used_time: float = 0.0
        self._cache_duration: int = 300 # 5 minutes
        # --- End Cache Tracking State ---

        logger.info(f"ModelManager initialized (device={self._device}, model_cache_size={self._max_models}, cache_duration={self._cache_duration}s)")

        # Add last health check timestamp
        self._last_health_check = time.time()
        self._health_check_interval = 20  # Check every 20 seconds (reduced from 60)

    def check_system_memory(self):
        """Check system (CPU) memory usage and return a warning if it's too high"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent > 85:
                logger.warning(f"System memory usage is high: {memory_percent:.1f}%")
                return False
            elif memory_percent > 75:
                logger.info(f"System memory usage is elevated: {memory_percent:.1f}%")

            return True
        except ImportError:
            logger.warning("psutil not available - cannot check system memory")
            return True
        except Exception as e:
            logger.error(f"Error checking system memory: {str(e)}")
            return True

    def check_gpu_health(self):
        """Check GPU memory usage and force cleanup if necessary"""
        # First check system memory
        system_healthy = self.check_system_memory()

        if self._device != "cuda":
            return system_healthy  # Only GPU checks applicable for CUDA devices

        try:
            current_time = time.time()
            # Only run check if interval has passed
            if current_time - self._last_health_check < self._health_check_interval:
                return True

            self._last_health_check = current_time

            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_memory = total_memory - memory_allocated

            # Only log if usage is high
            if memory_allocated / total_memory > 0.7:
                logger.info(f"GPU Health Check: {memory_allocated:.2f}GB/{total_memory:.2f}GB ({memory_allocated/total_memory:.2%})")

            # Check if we're approaching a dangerous threshold (e.g., 80% usage instead of 90%)
            memory_threshold = total_memory * 0.8
            if memory_allocated > memory_threshold:
                logger.warning(f"GPU memory usage exceeds threshold ({memory_allocated:.2f}GB > {memory_threshold:.2f}GB)")
                # --- Do not force cleanup based on threshold alone in this version ---
                # self._force_memory_cleanup()
                # return False  # Indicate unhealthy state
                # --- ---

            return True  # Healthy
        except Exception as e:
            logger.error(f"Error checking GPU health: {str(e)}")
            return False  # Consider unhealthy on error

    def _force_memory_cleanup(self):
        """Force aggressive memory cleanup"""
        logger.info("🧹 Performing aggressive memory cleanup")

        try:
            # Unload all models
            self._unload_all_models()

            # Force CUDA cache flush
            if self._device == "cuda":
                # Multiple passes of cleanup
                for _ in range(3):
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()  # Wait for all CUDA operations to complete

                # Print memory state after cleanup
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"🧹 Memory after cleanup: {memory_allocated:.2f}GB")

            # Try to force Python to release more memory
            import sys
            try:
                sys.set_asyncgen_hooks(firstiter=lambda gen: None, finalizer=lambda gen: None)
            except Exception:
                pass

            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()

            # Try to clean up any lingering references
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj):
                        obj.detach().cpu()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Error during forced cleanup: {str(e)}")

    def load_sd_pipeline(self, model_key: str) -> StableDiffusion3Pipeline:
        """Load SD 3.5 with proper configuration"""
        model_config = AVAILABLE_MODELS[model_key]
        model_repo = model_config["repo"]  # Use the repo field from config, not id

        # First check if we have a local path for this model
        model_path = os.path.join(current_app.config["MODELS_PATH"], model_key)
        if os.path.exists(model_path):
            logger.info(f"Loading SD 3.5 model from local path: {model_path}")

            # Configure 4-bit quantization
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            # Load transformer with quantization
            logger.debug(f"Loading transformer from local path: {model_path}/transformer")
            transformer = SD3Transformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.float16,
                local_files_only=True  # Enforce using only local files
            )

            # Load full pipeline
            logger.debug(f"Loading full SD3 pipeline from local path: {model_path}")
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                transformer=transformer,
                torch_dtype=torch.float16,
                local_files_only=True  # Enforce using only local files
            )
        else:
            # If local path doesn't exist, try using the repo
            logger.info(f"Loading SD 3.5 model from HuggingFace: {model_repo}")

            # Configure 4-bit quantization
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            # Load transformer with quantization
            transformer = SD3Transformer2DModel.from_pretrained(
                model_repo,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.float16,
                cache_dir=model_path
            )

            # Load full pipeline
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_repo,
                transformer=transformer,
                torch_dtype=torch.float16,
                cache_dir=model_path
            )

        # Enable CPU offloading as primary memory optimization
        pipeline.enable_model_cpu_offload()
        logger.debug("Enabled CPU offloading for SD 3.5 model")

        return pipeline

    def _fix_model_files(self, model_path: str, model_key: str) -> None:
        """
        Fix common issues with model files, like creating symlinks for differently named files
        """
        model_path = Path(model_path)
        if not model_path.exists():
            return

        logger.debug(f"Checking model directory structure for {model_key}")

        # Check for common VAE issues
        vae_dir = model_path / "vae"
        if vae_dir.exists():
            # Check for fp16 version without standard version
            fp16_file = vae_dir / "diffusion_pytorch_model.fp16.safetensors"
            standard_file = vae_dir / "diffusion_pytorch_model.safetensors"

            if fp16_file.exists() and not standard_file.exists():
                logger.info(f"Creating symlink from {fp16_file.name} to {standard_file.name}")
                try:
                    os.symlink(fp16_file.name, standard_file)
                    logger.info(f"Created symlink for VAE safetensors file")
                except Exception as e:
                    logger.error(f"Failed to create symlink: {str(e)}")

            # Also check for bin version
            bin_file = vae_dir / "diffusion_pytorch_model.bin"
            if not bin_file.exists() and standard_file.exists():
                logger.info(f"Creating symlink from {standard_file.name} to {bin_file.name}")
                try:
                    os.symlink(standard_file.name, bin_file)
                    logger.info(f"Created symlink for VAE bin file")
                except Exception as e:
                    logger.error(f"Failed to create symlink: {str(e)}")

        # Add more fixes here as needed

    def get_model(self, model_key: str) -> Optional[Union[FluxPipeline, DiffusionPipeline, LTXPipeline, WanPipeline]]:
        """Get a model by its key, loading it if necessary"""
        with self._model_lock: # Ensure thread-safe model loading/unloading
            model_config = AVAILABLE_MODELS.get(model_key)
            if not model_config:
                logger.error(f"Model configuration for '{model_key}' not found.")
                raise ModelLoadError(f"Model configuration for '{model_key}' not found.")

            # === ADDED/ENHANCED CHECK FOR ALL API MODELS ===
            # If it's an API model (any type), it's not loaded locally by this function.
            if model_config.get('source') == 'huggingface_api':
                logger.info(f"API model '{model_key}' (type: {model_config.get('type', 'unknown')}) requested in get_model. API models are handled by dedicated API call methods. Returning None.")
                return None
            # === END ADDED/ENHANCED CHECK ===

            if model_key in self._loaded_models:
                logger.info(f"Model {model_key} found in cache.")
                return self._loaded_models[model_key]

            # --- Check Cache ---
            if (model_key == self._currently_loaded_model_key and
                model_key in self._loaded_models and
                (time.time() - self._last_used_time < self._cache_duration)):

                try:
                    # Verify model is still valid (basic check)
                    model = self._loaded_models[model_key]
                    if hasattr(model, 'device'):
                        logger.info(f"Using cached model: {model_key} (last used {time.time() - self._last_used_time:.1f}s ago)")
                        self._last_used_time = time.time() # Update last used time on access
                        return model
                    else:
                        logger.warning(f"Cached model {model_key} seems invalid (no device attribute), reloading...")
                except Exception as e:
                    logger.warning(f"Error accessing cached model {model_key}, reloading... Error: {str(e)}")
            # --- End Cache Check ---

            # --- Load or Reload Model ---
            # If cache miss, different model, or timeout expired, unload existing and load new
            if self._currently_loaded_model_key:
                logger.info(f"Unloading previous model ({self._currently_loaded_model_key}) to load {model_key}")
            else:
                logger.info(f"Loading model: {model_key} (no model currently cached)")

            self._unload_all_models() # Clears cache tracking vars too

            # Additional cleanup to ensure maximum memory is available
            if self._device == "cuda":
                for _ in range(2):
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()

                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.debug(f"GPU memory before loading: {memory_allocated:.2f}GB/{total_memory:.2f}GB")

            try:
                model_path = os.path.join(current_app.config["MODELS_PATH"], model_key)
                model_config = AVAILABLE_MODELS[model_key]
                model_type = model_config.get("type", "").lower()

                # Check and fix model file structure
                self._fix_model_files(model_path, model_key)

                # --- Define Memory Optimization Helper (for Image models) --- #
                # Define this *before* the model type check so it's available
                def apply_memory_optimizations(pipe):
                    # Use CPU offloading as the primary memory optimization strategy
                    pipe.enable_model_cpu_offload()
                    logger.debug("Enabled CPU offloading to minimize GPU memory usage")

                    # Print memory usage after optimizations
                    if self._device == "cuda":
                        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                        logger.debug(f"GPU memory after optimizations: {memory_allocated:.2f}GB")

                    return pipe

                # --- ADDED: Handle LTX GGUF T2V Model First --- #
                if model_config.get("source") == "gguf_url" and model_type == "t2v":
                    logger.info(f"Loading LTX-Video GGUF model: {model_key}")
                    if GGUFQuantizationConfig is None:
                        raise ModelLoadError("GGUFQuantizationConfig is required for LTX GGUF models but was not found in diffusers.")

                    gguf_files = list(Path(model_path).glob("*.gguf"))
                    if not gguf_files:
                        raise ModelLoadError(f"No .gguf file found in model directory: {model_path}")
                    if len(gguf_files) > 1:
                        logger.warning(f"Multiple .gguf files found in {model_path}, using the first one: {gguf_files[0]}")

                    gguf_file_path = gguf_files[0]
                    logger.info(f"Loading LTX Transformer from GGUF: {gguf_file_path}")

                    # Determine compute dtype based on availability
                    # compute_dtype = torch.bfloat16 if is_torch_available() else torch.float16
                    # --- Align with Reference Code: Use bfloat16 directly ---
                    dtype_to_use = torch.bfloat16
                    logger.info(f"Using dtype: {dtype_to_use}")
                    # --- End Alignment ---
                    quant_config = GGUFQuantizationConfig(compute_dtype=dtype_to_use)

                    transformer = LTXVideoTransformer3DModel.from_single_file(
                        str(gguf_file_path),
                        quantization_config=quant_config,
                        torch_dtype=dtype_to_use, # Use aligned dtype for loading
                    )

                    logger.info("Loading LTX Pipeline base...")
                    # Load the base LTXPipeline first
                    pipe = LTXPipeline.from_pretrained(
                        "Lightricks/LTX-Video", # Base pipeline definition
                        transformer=transformer,
                        torch_dtype=dtype_to_use, # Use aligned dtype for pipeline
                        local_files_only=False # Allow downloading other components if needed
                    )
                    pipe = apply_memory_optimizations(pipe) # Apply optimizations
                    logger.info(f"Loaded LTX GGUF model ({model_key}) with CPU offloading")
                # --- Handle Video Model Type --- #
                elif model_type == "i2v":
                    logger.info(f"Loading Image-to-Video (I2V) model: {model_key}")

                    # --- Check if it's the LTX GGUF model --- #
                    # NOTE: Assuming the same key ('LTX-Video') is used for GGUF I2V for now.
                    # This might need adjustment based on actual config key.
                    is_ltx_gguf_i2v = model_config.get("source") == "gguf_url"

                    if is_ltx_gguf_i2v:
                        logger.warning("Attempting EXPERIMENTAL load of LTXImageToVideoPipeline with GGUF transformer.")
                        if GGUFQuantizationConfig is None:
                            raise ModelLoadError("GGUFQuantizationConfig is required but was not found.")

                        gguf_files = list(Path(model_path).glob("*.gguf"))
                        if not gguf_files:
                            raise ModelLoadError(f"No .gguf file found for LTX I2V: {model_path}")
                        gguf_file_path = gguf_files[0]

                        dtype_to_use = torch.bfloat16
                        quant_config = GGUFQuantizationConfig(compute_dtype=dtype_to_use)

                        logger.info(f"Loading LTX Transformer from GGUF: {gguf_file_path}")
                        transformer = LTXVideoTransformer3DModel.from_single_file(
                            str(gguf_file_path),
                            quantization_config=quant_config,
                            torch_dtype=dtype_to_use,
                        )

                        logger.info("Loading LTXImageToVideoPipeline base and injecting GGUF transformer...")
                        try:
                            # Attempt to use LTXImageToVideoPipeline with the GGUF transformer
                            pipe = LTXImageToVideoPipeline.from_pretrained(
                                "Lightricks/LTX-Video", # Base pipeline definition
                                transformer=transformer, # Pass GGUF transformer
                                torch_dtype=dtype_to_use,
                                local_files_only=False
                            )
                        except TypeError as te:
                            logger.error(f"Failed to load LTXImageToVideoPipeline with GGUF transformer: {te}")
                            logger.error("This likely means LTXImageToVideoPipeline doesn't accept a custom transformer this way.")
                            raise ModelLoadError("Incompatible GGUF transformer for LTXImageToVideoPipeline") from te

                        pipe = apply_memory_optimizations(pipe) # Apply optimizations
                        logger.info(f"Loaded LTX GGUF for I2V ({model_key}) with CPU offloading (Experimental)")

                    else:
                        # --- Existing Wan I2V Logic --- #
                        video_dtype = torch.bfloat16 if self._dtype == torch.bfloat16 else torch.float16
                        vae_dtype = torch.float32
                        encoder_dtype = torch.float32

                        logger.debug(f"Loading Wan video components: VAE ({vae_dtype}), Image Encoder ({encoder_dtype}), Pipeline ({video_dtype})")

                        image_encoder = CLIPVisionModel.from_pretrained(
                            model_config['repo'],
                            subfolder="image_encoder",
                            torch_dtype=encoder_dtype,
                            local_files_only=False, # Allow download if missing
                            cache_dir=model_path
                        )
                        vae = AutoencoderKLWan.from_pretrained(
                            model_config['repo'], # Use repo ID like other components
                            subfolder="vae",
                            torch_dtype=vae_dtype,
                            local_files_only=False, # Allow download if missing
                            cache_dir=model_path
                        )
                        pipe = WanImageToVideoPipeline.from_pretrained(
                            model_config['repo'],
                            vae=vae,
                            image_encoder=image_encoder,
                            torch_dtype=video_dtype,
                            local_files_only=False, # Allow download if missing
                            cache_dir=model_path
                        )
                        # --- Skip CPU offloading for Wan I2V to maximize speed ---
                        pipe.to(self._device) # Move directly to the target device
                        logger.info(f"Loaded Wan I2V model ({model_key}) directly to {self._device} (no CPU offload) using dtype: {video_dtype}")
                        # --- ---
                elif model_type == "t2v":
                    logger.info(f"Loading Text-to-Video (T2V) model: {model_key}")
                    # --- Use WanPipeline and load VAE separately --- #
                    t2v_dtype = torch.bfloat16 if self._dtype == torch.bfloat16 else torch.float16
                    vae_dtype = torch.float32 # Use float32 for VAE stability

                    logger.debug(f"Loading T2V VAE ({vae_dtype}) and Pipeline ({t2v_dtype})")

                    vae = AutoencoderKLWan.from_pretrained(
                        model_path,
                        subfolder="vae",
                        torch_dtype=vae_dtype,
                        local_files_only=True,
                        cache_dir=model_path
                    )
                    pipe = WanPipeline.from_pretrained(
                        model_path,
                        vae=vae,
                        torch_dtype=t2v_dtype,
                        local_files_only=True,
                        cache_dir=model_path
                    )
                    # --- End WanPipeline specific loading ---

                    # --- Apply Memory Optimization --- #
                    pipe = apply_memory_optimizations(pipe)
                    logger.info(f"Loaded T2V model ({model_key}) with CPU offloading")
                    # --- End Memory Optimization --- #

                # --- Handle Image Model Types (Existing Logic) ---
                else:
                    # Check if this is an API model
                    if model_config.get('source') == 'huggingface_api':
                        logger.info(f"Skipping model loading for API model: {model_key}")
                        return None  # API models don't need to be loaded as pipelines

                    # Common configuration for all image models
                    load_config = {
                        "torch_dtype": torch.float16,  # Use float16 for better memory efficiency
                        "local_files_only": True,
                    }

                    # Dynamic model loading based on model_key or model_type
                    if model_key == "sd-3.5" or model_type == "sd3":
                        # SD 3.5 has special handling with transformer
                        pipe = self.load_sd_pipeline(model_key)
                    elif model_key == "sana-sprint":
                        logger.debug(f"Loading SanaSprint model from local path: {model_path}")
                        pipe = SanaSprintPipeline.from_pretrained(
                            model_path,
                            torch_dtype=self._dtype,
                            local_files_only=True
                        )
                        # --- Skip CPU offloading for SanaSprint to maximize speed ---
                        pipe.to(self._device) # Move directly to the target device
                        logger.info(f"Loaded SanaSprint model ({model_key}) directly to {self._device} (no CPU offload) using dtype: {self._dtype}")
                        # --- ---
                    elif "flux" in model_key.lower() or model_type == "flux":
                        # Apply Flux profile to any model key containing "flux"
                        pipe = FluxPipeline.from_pretrained(
                            model_path,
                            **load_config
                        )
                        pipe = apply_memory_optimizations(pipe)
                        logger.info(f"Loaded FLUX model ({model_key}) with CPU offloading")
                    elif model_key == "sd-xl" or model_type == "sdxl":
                        # Verify model path exists and load from local path
                        if os.path.exists(model_path):
                            logger.debug(f"Loading SDXL model from local path: {model_path}")
                            pipe = DiffusionPipeline.from_pretrained(
                                model_path,
                                **load_config
                            )
                        else:
                            logger.debug(f"Loading SDXL model from repo: {model_config['repo']}")
                            pipe = DiffusionPipeline.from_pretrained(
                                model_config['repo'],
                                cache_dir=model_path,
                                **load_config
                            )
                        pipe = apply_memory_optimizations(pipe)
                        logger.info(f"Loaded SDXL model with CPU offloading")
                    else:
                        # Default loading for other model types
                        if os.path.exists(model_path):
                            logger.debug(f"Loading generic model from local path: {model_path}")
                            pipe = DiffusionPipeline.from_pretrained(
                                model_path,
                                **load_config
                            )
                        else:
                            logger.debug(f"Loading generic model from repo: {model_config['repo']}")
                            pipe = DiffusionPipeline.from_pretrained(
                                model_config['repo'],
                                cache_dir=model_path,
                                **load_config
                            )
                        pipe = apply_memory_optimizations(pipe)
                        logger.info(f"Loaded generic diffusion model with CPU offloading")

                if pipe is None:
                    raise ValueError(f"Failed to load model {model_key}")

                # Verify the model is on the correct device
                if hasattr(pipe, 'device'):
                    logger.debug(f"Model device: {pipe.device}")

                # Update memory tracking after loading
                if self._device == "cuda":
                    torch.cuda.synchronize()
                    memory_allocated = torch.cuda.memory_allocated() / (1024**2)
                    memory_reserved = torch.cuda.memory_reserved() / (1024**2)
                    logger.debug(f"Post-load GPU Memory: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB")
                    torch.cuda.empty_cache()
                    gc.collect()

                # Update cache tracking AFTER successful load
                self._currently_loaded_model_key = model_key
                self._last_used_time = time.time()
                self._loaded_models[model_key] = pipe
                return pipe

            except Exception as e:
                print(f"\n❌ Error loading model {model_key}: {str(e)}")
                sys.stdout.flush()
                raise ModelLoadError(f"Failed to load model {model_key}: {str(e)}")

    def _unload_all_models(self):
        """Unload all models and clear GPU memory"""
        logger.info("Unloading all models to free memory")
        for key in list(self._loaded_models.keys()):
            logger.debug(f"Unloading {key}")
            del self._loaded_models[key]

        # Reset cache tracking
        self._currently_loaded_model_key = None
        self._last_used_time = 0.0

        if self._device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            logger.debug(f"GPU memory after cleanup: {memory_allocated:.2f}GB")

    def generate_image(self, model_key: str, prompt: str, **kwargs) -> torch.Tensor:
        """Generate an image using the specified model"""
        try:
            # Enhanced logging
            model_info = AVAILABLE_MODELS.get(model_key, {})
            model_desc = model_info.get("description", "Unknown model")
            model_type = model_info.get("type", "image")
            print(f"\n🖼️ GENERATING IMAGE with model: {model_key} - {model_desc}")
            print(f"   • Model Type: {model_type}")
            print(f"   • Prompt: {prompt[:100]}" + ('...' if len(prompt) > 100 else ''))
            if "negative_prompt" in kwargs and kwargs["negative_prompt"]:
                print(f"   • Negative: {kwargs['negative_prompt'][:100]}" + ('...' if len(kwargs['negative_prompt']) > 100 else ''))

            # Log key parameters
            steps = kwargs.get("num_inference_steps", "default")
            guidance = kwargs.get("guidance_scale", "default")
            dimensions = f"{kwargs.get('width', 'default')}x{kwargs.get('height', 'default')}"
            print(f"   • Parameters: Steps={steps}, Guidance={guidance}, Size={dimensions}")
            sys.stdout.flush()
            # End enhanced logging

            # Get the model
            pipe = self.get_model(model_key)
            if pipe is None:
                raise ModelLoadError(f"Failed to get model {model_key}")

            # --- Extract and Prepare Pipeline Arguments ---
            pipe_args = {}
            pipe_args["prompt"] = prompt

            # Extract known arguments from kwargs (job settings)
            # Use model's step_config for defaults where applicable
            model_config = AVAILABLE_MODELS.get(model_key, {})
            step_config = model_config.get("step_config", {})

            # Steps (handle sana-sprint specifically)
            if model_key == "sana-sprint":
                pipe_args["num_inference_steps"] = 2
                logger.info("Forcing num_inference_steps=2 for sana-sprint model.")
            elif "fixed_steps" in step_config:
                 pipe_args["num_inference_steps"] = step_config["fixed_steps"]
            else:
                 default_steps = step_config.get("steps", {}).get("default", 30)
                 pipe_args["num_inference_steps"] = kwargs.get("num_inference_steps", default_steps)

            # Guidance
            default_guidance = 7.5
            pipe_args["guidance_scale"] = kwargs.get("guidance_scale", default_guidance)

            # Dimensions
            default_height = 1024
            default_width = 1024
            pipe_args["height"] = kwargs.get("height", default_height)
            pipe_args["width"] = kwargs.get("width", default_width)

            # Max sequence length (if applicable to pipeline)
            # Note: Not all pipelines use this. Pass only if needed or part of default args.
            # pipe_args["max_sequence_length"] = kwargs.get("max_sequence_length", 512)

            # Handle negative prompt based on model type
            negative_prompt = kwargs.get("negative_prompt", None)
            # Check if the specific pipeline type accepts negative_prompt (example for SD3)
            if negative_prompt and model_key == "sd-3.5":
                pipe_args["negative_prompt"] = negative_prompt
            # --- End Argument Preparation ---

            logger.info(f"Generating image with {model_key}: {prompt[:50]}... Args: { {k:v for k,v in pipe_args.items() if k != 'prompt'} }")

            # Create generator on appropriate device
            if self._device == "mps":
                generator = torch.Generator("cpu").manual_seed(
                    kwargs.get("seed", None) or torch.randint(1, 9999999, (1,)).item()
                )
            else:
                generator = torch.Generator(device=self._device).manual_seed(
                    kwargs.get("seed", None) or torch.randint(1, 9999999, (1,)).item()
                )

            # Log generation start
            print(f"\n🚀 Starting image generation with {model_key} ({pipe_args['num_inference_steps']} steps)")
            sys.stdout.flush()

            # Standard generation for all models - no callbacks
            result = pipe(
                **pipe_args # Pass prepared arguments directly
            ).images[0]

            # Success!
            print(f"✅ Successfully generated image with model {model_key}")
            sys.stdout.flush()
            logger.info(f"Successfully generated image with model {model_key}")
            return result

        except Exception as e:
            error_msg = f"Failed to generate image: {str(e)}"
            logger.error(error_msg)
            # Force cleanup on error
            self._force_memory_cleanup()
            raise ModelGenerationError(error_msg)

    def generate_image_from_text(self, model_key: str, prompt: str, **kwargs) -> Tuple[str, Dict]:
        """Generate an image from text using the specified model"""
        try:
            # Enhanced logging
            model_info = AVAILABLE_MODELS.get(model_key, {})
            model_desc = model_info.get("description", "Unknown model")
            print(f"\n🖼️ GENERATING IMAGE with model: {model_key} - {model_desc}")
            print(f"   • Prompt: {prompt[:100]}" + ('...' if len(prompt) > 100 else ''))
            
            # Check if this is an API model
            if model_info.get('source') == 'huggingface_api':
                # Get provider from options_json
                provider = model_info.get('step_config', {}).get('provider')
                if not provider:
                    raise ValueError(f"No provider specified for API model {model_key}")
                
                # Use InferenceClient for API models
                from huggingface_hub import InferenceClient
                client = InferenceClient(token=os.getenv('HF_TOKEN'))
                
                # Prepare parameters
                params = {
                    "prompt": prompt,
                    "negative_prompt": kwargs.get("negative_prompt", ""),
                    "width": kwargs.get("width", 1024),
                    "height": kwargs.get("height", 1024),
                    "num_inference_steps": kwargs.get("num_inference_steps", 30),
                    "guidance_scale": kwargs.get("guidance_scale", 7.5)
                }
                
                print(f"   • Parameters: Steps={params['num_inference_steps']}, Guidance={params['guidance_scale']}, Size={params['width']}x{params['height']}")
                print(f"   • Using provider: {provider}")
                sys.stdout.flush()
                
                # Generate image using API
                print(f"\n🚀 Starting image generation with {model_key} via {provider}")
                sys.stdout.flush()
                
                # Get the model ID from the repo field
                model_id = model_info.get('repo')
                if not model_id:
                    raise ValueError(f"No model ID specified for API model {model_key}")
                
                # Generate image using InferenceClient
                result = client.text_to_image(
                    model=model_id,
                    **params
                )
                
                print(f"✅ Successfully generated image with model {model_key} via {provider}")
                sys.stdout.flush()
                
                # Convert PIL image to base64 string
                from io import BytesIO
                import base64
                buffered = BytesIO()
                result.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Return base64 image and metadata
                return img_str, {
                    "model": model_key,
                    "provider": provider,
                    "parameters": params
                }
            
            # For non-API models, use the existing pipeline approach
            sys.stdout.flush()
            # End enhanced logging

            # Extract common parameters
            negative_prompt = kwargs.get("negative_prompt", "")
            width = kwargs.get("width", 1024)
            height = kwargs.get("height", 1024)
            num_inference_steps = kwargs.get("num_inference_steps", 20)
            guidance_scale = kwargs.get("guidance_scale", 7.5)
            seed = kwargs.get("seed", None)
            output_folder = current_app.config.get('IMAGES_FOLDER', 'app/static/images')
            Path(output_folder).mkdir(parents=True, exist_ok=True) # Ensure output folder exists

            model_config = AVAILABLE_MODELS.get(model_key)
            if not model_config:
                raise ModelGenerationError(f"Model configuration not found for key: {model_key}")

            generated_image_pil = None
            final_width, final_height = width, height # Initialize with requested dimensions

            # --- Hugging Face API Integration --- # 
            if model_config.get('source') == 'huggingface_api':
                logger.info(f"Using Hugging Face API for model: {model_key}")
                if not self._hf_token and (model_config.get('step_config', {}).get('provider') is None or model_config.get('step_config', {}).get('provider') == "huggingface-inference-api"):
                    raise ModelGenerationError("HF_TOKEN is not configured. Cannot use API-based models.")
                if not self._replicate_api_key and model_config.get('step_config', {}).get('provider') == "replicate":
                    raise ModelGenerationError("REPLICATE_API_KEY is not configured. Cannot use Replicate API-based models.")

                provider = model_config.get('step_config', {}).get('provider')
                api_model_id = model_config.get('repo') # 'repo' field stores the API model identifier

                if not api_model_id:
                     raise ModelGenerationError(f"API model ID ('repo') missing in config for {model_key}.")

                logger.info(f"Attempting API call: Provider='{provider}', Model ID='{api_model_id}'")
                
                client = None
                try:
                    if provider == "replicate":
                        if not self._replicate_api_key:
                            raise ModelGenerationError("Replicate API key is missing. Cannot use Replicate provider.")
                        client = InferenceClient(provider="replicate", api_key=self._replicate_api_key)
                        logger.debug("Instantiated InferenceClient for Replicate with API key.")
                    elif provider == "huggingface-inference-api" or provider is None: # Default to HF
                        if not self._hf_token:
                            raise ModelGenerationError("Hugging Face token (HF_TOKEN) is missing. Cannot use Hugging Face provider.")
                        # Provider can be omitted for HF default, or explicitly set
                        client = InferenceClient(provider=provider if provider else None, token=self._hf_token)
                        logger.debug(f"Instantiated InferenceClient for Hugging Face provider='{provider if provider else 'default'}' with HF_TOKEN.")
                    elif provider == "fal-ai":
                        if not self._hf_token:
                            raise ModelGenerationError("HF_TOKEN is missing. Cannot use fal-ai provider (authenticated via HF Token).")
                        client = InferenceClient(provider="fal-ai", api_key=self._hf_token)
                        logger.debug("Instantiated InferenceClient for fal-ai with HF_TOKEN.")
                    else:
                        # Fallback for other potential providers, assuming token-based for now
                        # This part might need more specific handling if other key-based providers are added
                        logger.warning(f"Provider '{provider}' specified. Attempting with HF_TOKEN. This provider might require its own API key configuration.")
                        if not self._hf_token:
                             raise ModelGenerationError(f"HF_TOKEN is missing, cannot attempt API call for provider '{provider}'.")
                        client = InferenceClient(provider=provider, token=self._hf_token)

                    api_params = {
                        "prompt": prompt,
                        "model": api_model_id, # Pass model identifier to text_to_image
                        "negative_prompt": negative_prompt if negative_prompt else None,
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "seed": seed if seed is not None else "N/A (API or not set)",
                    }
                    logger.debug(f"Calling InferenceClient.text_to_image with params: {api_params}")
                    generated_image_pil = client.text_to_image(**api_params)
                    
                    if not isinstance(generated_image_pil, Image.Image):
                        logger.error(f"API for {model_key} did not return a PIL Image. Got: {type(generated_image_pil)}")
                        raise ModelGenerationError(f"Unexpected response type from API for model {model_key}.")
                    
                    final_width, final_height = generated_image_pil.size
                    logger.info(f"Image generated via API ({model_key}) successfully. Size: {final_width}x{final_height}")

                except Exception as e:
                    logger.error(f"Hugging Face API generation failed for {model_key} (Provider: {provider}, Model: {api_model_id}): {str(e)}")
                    import traceback # For more detailed error logging
                    logger.error(traceback.format_exc())
                    raise ModelGenerationError(f"API generation error for {model_key}: {str(e)}")
            
            # --- Local Model Generation --- #
            else:
                logger.info(f"Using local model for: {model_key}")
                # Ensure GPU health before loading/using local model
                if not self.check_gpu_health():
                    logger.error("GPU health check failed. Aborting image generation.")
                    raise ModelGenerationError("GPU is not healthy. Please check logs.")

                pipe = self.get_model(model_key)
                if pipe is None:
                    raise ModelLoadError(f"Failed to get model {model_key}")

                # Determine generator for reproducibility if seed is provided
                generator = torch.Generator(device=self._device).manual_seed(seed) if seed is not None else None

                # --- Specific handling for FLUX models --- # 
                if model_config.get("type") == "flux" or "flux" in model_key.lower():
                    logger.info(f"Using FLUX-specific pipeline settings for {model_key}")
                    # FLUX pipeline uses 'prompt' and directly takes guidance_scale, num_inference_steps
                    # It might not use negative_prompt in the same way or at all depending on the version.
                    # The base FLUX.1-schnell doesn't use negative_prompt in its example.
                    # For FLUX.1-dev, it's usually part of prompt engineering or specific variants.
                    flux_params = {
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale, # FLUX uses this
                        "generator": generator
                    }
                    # Only add negative_prompt if it's non-empty and the model might support it (heuristic)
                    if negative_prompt: # Add if provided, specific FLUX variants might use it
                         flux_params["negative_prompt"] = negative_prompt
                    
                    logger.debug(f"Calling FLUX pipeline with params: {flux_params}")
                    output = pipe(**flux_params)

                # --- Specific handling for SD3.5 models --- # 
                elif model_config.get("type") == "sd3" or "sd-3" in model_key.lower():
                    logger.info(f"Using SD3.5-specific pipeline settings for {model_key}")
                    sd3_params = {
                        "prompt": prompt, 
                        "negative_prompt": negative_prompt if negative_prompt else None,
                        "width": width, 
                        "height": height,
                        "num_inference_steps": num_inference_steps, 
                        "guidance_scale": guidance_scale,
                        "generator": generator
                    }
                    logger.debug(f"Calling SD3 pipeline with params: {sd3_params}")
                    output = pipe(**sd3_params)

                # --- Generic Diffusers Pipeline --- # 
                else:
                    logger.info(f"Using generic Diffusers pipeline settings for {model_key}")
                    pipe_params = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "generator": generator
                    }
                    logger.debug(f"Calling generic pipeline with params: {pipe_params}")
                    output = pipe(**pipe_params)

                generated_image_pil = output.images[0]
                final_width, final_height = generated_image_pil.size
                logger.info(f"Image generated locally ({model_key}) successfully. Size: {final_width}x{final_height}")

            # --- Save image and prepare metadata (common for both API and local) --- #
            if generated_image_pil:
                # Generate unique filename
                unique_id = uuid.uuid4()
                image_filename = f"{model_key.replace('/', '_')}_{unique_id}.png"
                image_path = Path(output_folder) / image_filename

                # Save the image
                generated_image_pil.save(image_path, "PNG")
                logger.info(f"Image saved to {image_path}")

                # Prepare metadata
                metadata_full = {
                    "model_key": model_key,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "width": final_width, # Use actual dimensions of generated image
                    "height": final_height, # Use actual dimensions of generated image
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed if seed is not None else "N/A (API or not set)",
                    "source": model_config.get('source', 'unknown'), # Add source to metadata
                    "provider": model_config.get('step_config', {}).get('provider') if model_config.get('source') == 'huggingface_api' else None,
                    "image_path": str(image_path),
                    "filename": image_filename,
                    "generation_time": time.time() - kwargs.get("start_time", time.time())
                }
                # Remove provider from metadata if it's None (for local models)
                if metadata_full["provider"] is None:
                    del metadata_full["provider"]

                return str(image_path), metadata_full
            else:
                raise ModelGenerationError(f"Image generation failed for {model_key}, no image produced.")

        except ModelLoadError as mle: # Specific exception from get_model
            logger.error(f"Model loading failed during image generation for {model_key}: {str(mle)}")
            self._force_memory_cleanup() # Attempt cleanup on load failure
            raise ModelGenerationError(f"Failed to load model {model_key} for generation: {str(mle)}")
        except ModelGenerationError: # Re-raise if it's already one of ours
            raise
        except Exception as e:
            logger.error(f"Unexpected error during image generation for {model_key}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Attempt cleanup for unexpected errors too, as GPU state might be unstable
            if model_config and model_config.get('source') != 'huggingface_api':
                 self._force_memory_cleanup()
            raise ModelGenerationError(f"An unexpected error occurred while generating image with {model_key}: {str(e)}")

    def generate_image_to_video(self, model_key: str, video_prompt: str, source_image_path: str, **kwargs):
        model_config = AVAILABLE_MODELS.get(model_key)
        if not model_config:
            raise ModelLoadError(f"Model {model_key} not found in configuration.")

        # All I2V is now assumed to be API based
        if model_config.get("source") != "huggingface_api":
            # This condition should ideally not be met if load_models ensures only API i2v are considered runnable
            logger.error(f"Model {model_key} is configured for I2V but not as an API model. Local I2V is deprecated.")
            raise ModelGenerationError(f"Local Image-to-Video for {model_key} is no longer supported. Configure as API model.")

        # API-based I2V generation logic (existing code)
        try:
            logger.info(f"🎬 Starting Image-to-Video generation (I2V) with API model: {model_key}")
            # ... (existing API call logic, parameter extraction, client init, etc.)
            # Ensure all necessary parameters are passed from kwargs and model_config.options
            api_token_to_use = self._hf_token
            api_token_env_var = model_config.get("options", {}).get("api_token_env_var")
            if api_token_env_var:
                custom_token = os.getenv(api_token_env_var)
                if custom_token:
                    api_token_to_use = custom_token
                else:
                    logger.warning(f"api_token_env_var '{api_token_env_var}' specified for model {model_key} but env var not set or empty.")
        
            # Get the provider from model_config options to be used in the check
            current_provider_from_config = model_config.get("options", {}).get("provider")

            # ***** DEBUGGING LOGS START *****
            logger.info(f"DEBUG I2V/T2V: For model_key='{model_key}', before 'API token not configured' check:")
            logger.info(f"DEBUG I2V/T2V:   self._hf_token is: {'SET and VALID' if self._hf_token else 'NOT SET or EMPTY'}")
            logger.info(f"DEBUG I2V/T2V:   model_config.get('options', {{}}).get('api_token_env_var') = '{model_config.get('options', {}).get('api_token_env_var')}'")
            logger.info(f"DEBUG I2V/T2V:   api_token_to_use (resolved) is: {'SET and VALID' if api_token_to_use else 'NOT SET or EMPTY'}")
            logger.info(f"DEBUG I2V/T2V:   Provider from model_config.options = '{current_provider_from_config}'")
            logger.info(f"DEBUG I2V/T2V:   Condition (not api_token_to_use): {not api_token_to_use}")
            logger.info(f"DEBUG I2V/T2V:   Condition (not current_provider_from_config == 'fal-ai'): {not current_provider_from_config == 'fal-ai'}")
            # ***** DEBUGGING LOGS END *****

            # This is the problematic check
            if not api_token_to_use and not current_provider_from_config == "fal-ai":
                logger.error(f"DEBUG I2V/T2V: Raising 'API token not configured' error. Both conditions TRUE. api_token_to_use:'{api_token_to_use}', provider:'{current_provider_from_config}'")
                raise ModelGenerationError(f"An API token (HF_TOKEN or custom) is not configured for API model {model_key} and it's not a provider like 'fal-ai' that uses a separate key.")

            # Fal.ai provider for I2V and T2V (using HF_TOKEN)
            # This 'provider' variable is for the switch-case like logic below, distinct from current_provider_from_config used in the check above.
            provider = model_config.get("options", {}).get("provider") 
            hf_model_repo_id = model_config.get("repo")
            if not hf_model_repo_id:
                raise ModelGenerationError(f"API model ID (repo) not configured for {model_key}.")

            client = None
            if provider == "fal-ai":
                if not self._hf_token:
                    raise ModelGenerationError(f"HF_TOKEN not configured for model {model_key} using fal-ai provider (authenticated via HF Token).")
                client = InferenceClient(provider="fal-ai", api_key=self._hf_token)
                logger.info(f"Using fal-ai provider for I2V model {model_key} with HF_TOKEN.")
            else:
                token_to_use = self._hf_token 
                token_source_log = "default HF_TOKEN"
                if api_token_env_var:
                    custom_token = os.getenv(api_token_env_var)
                    if custom_token:
                        token_to_use = custom_token
                        token_source_log = f"token from {api_token_env_var}"
                    else:
                        logger.warning(f"Environment variable {api_token_env_var} for {model_key} not found, falling back to HF_TOKEN.")
                if not token_to_use:
                    raise ModelGenerationError(f"An API token ({token_source_log}) is not configured for API model {model_key}.")
                client = InferenceClient(token=token_to_use)
                logger.info(f"Using {token_source_log} for I2V model {model_key}.")

            logger.info(f"Attempting I2V with API model: {model_key}, provider: {provider or 'default HuggingFace'}")

            pil_image = load_image(source_image_path) # Assuming load_image is defined and returns PIL Image
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_data_url = f"data:image/png;base64,{img_base64}"

            payload_input = {
                "prompt": video_prompt,
                "image_url": img_data_url,
            }
            if kwargs.get("negative_prompt"): payload_input["negative_prompt"] = kwargs["negative_prompt"]
            if kwargs.get("num_frames") is not None: payload_input["num_frames"] = kwargs["num_frames"]
            if kwargs.get("fps") is not None: payload_input["fps"] = kwargs["fps"]
            if kwargs.get("motion_bucket_id") is not None: payload_input["motion_bucket_id"] = kwargs["motion_bucket_id"]
            if kwargs.get("noise_aug_strength") is not None: payload_input["noise_aug_strength"] = kwargs["noise_aug_strength"]
            if kwargs.get("seed") is not None: payload_input["seed"] = kwargs["seed"]

            request_payload = {"input": payload_input}

            logger.debug(f"Sending POST request. Model: {hf_model_repo_id}, Payload: {request_payload}")
            video_bytes = client.post(model=hf_model_repo_id, json=request_payload)

            generation_time = time.time() - kwargs.get("start_time", time.time())
            logger.info(f"✅ I2V (API) for {model_key} completed in {generation_time:.2f}s, output {len(video_bytes)} bytes.")
            return video_bytes
        except Exception as e:
            logger.error(f"Error during I2V API generation for {model_key}: {e}", exc_info=True)
            error_msg = f"Failed to generate I2V video with {model_key}: {e}"
            raise ModelGenerationError(error_msg) from e

    def generate_text_to_video(self, model_key: str, prompt: str, **kwargs):
        model_config = AVAILABLE_MODELS.get(model_key)
        if not model_config:
            raise ModelLoadError(f"Model {model_key} not found in configuration.")

        # All T2V is now assumed to be API based
        if model_config.get("source") != "huggingface_api":
            # This condition should ideally not be met if load_models ensures only API t2v are considered runnable
            logger.error(f"Model {model_key} is configured for T2V but not as an API model. Local T2V is deprecated.")
            raise ModelGenerationError(f"Local Text-to-Video for {model_key} is no longer supported. Configure as API model.")

        # API-based T2V generation logic (existing code)
        try:
            logger.info(f"🎬 Starting Text-to-Video generation (T2V) with API model: {model_key}")
            # ... (existing API call logic, parameter extraction, client init, etc.)
            # Ensure all necessary parameters are passed from kwargs and model_config.options
            hf_model_repo_id = model_config.get("repo")
            if not hf_model_repo_id:
                raise ModelGenerationError(f"API model ID (repo) not configured for {model_key}.")

            client = InferenceClient(token=self._hf_token)

            api_params = {}
            options = model_config.get("options", {})
            if kwargs.get("num_frames") is not None: api_params["num_frames"] = kwargs["num_frames"]
            elif options.get("num_frames") is not None: api_params["num_frames"] = options["num_frames"]
            if kwargs.get("fps") is not None: api_params["num_frames_per_second"] = kwargs["fps"]
            elif options.get("fps") is not None: api_params["num_frames_per_second"] = options["fps"]
            if kwargs.get("width") is not None: api_params["width"] = kwargs["width"]
            elif options.get("width") is not None: api_params["width"] = options["width"]
            if kwargs.get("height") is not None: api_params["height"] = kwargs["height"]
            elif options.get("height") is not None: api_params["height"] = options["height"]
            if kwargs.get("guidance_scale") is not None: api_params["guidance_scale"] = kwargs["guidance_scale"]
            elif options.get("guidance_scale") is not None: api_params["guidance_scale"] = options["guidance_scale"]
            if kwargs.get("seed") is not None: api_params["seed"] = kwargs["seed"]

            provider = model_config.get("options", {}).get("provider")
            client = None

            if provider == "fal-ai":
                if not self._hf_token:
                    raise ModelGenerationError(f"HF_TOKEN not configured for model {model_key} using fal-ai provider (authenticated via HF Token).")
                client = InferenceClient(provider="fal-ai", api_key=self._hf_token)
                logger.info(f"Using fal-ai provider for T2V model {model_key} with HF_TOKEN.")
            else:
                api_token_env_var = model_config.get("options", {}).get("api_token_env_var")
                token_to_use = self._hf_token 
                token_source_log = "default HF_TOKEN"
                if api_token_env_var:
                    custom_token = os.getenv(api_token_env_var)
                    if custom_token:
                        token_to_use = custom_token
                        token_source_log = f"token from {api_token_env_var}"
                    else:
                        logger.warning(f"Environment variable {api_token_env_var} for {model_key} not found, falling back to HF_TOKEN.")
                if not token_to_use:
                    raise ModelGenerationError(f"An API token ({token_source_log}) is not configured for API model {model_key}.")
                client = InferenceClient(token=token_to_use)
                logger.info(f"Using {token_source_log} for T2V model {model_key}.")

            logger.debug(f"Text-to-Video API call for {model_key} ({hf_model_repo_id}) with params: {api_params}")
            video_bytes = client.text_to_video(
                prompt=prompt,
                model=hf_model_repo_id, 
                negative_prompt=kwargs.get("negative_prompt"),
                **api_params
            )
            generation_time = time.time() - kwargs.get("start_time", time.time())
            logger.info(f"✅ T2V (API) for {model_key} completed in {generation_time:.2f}s, output {len(video_bytes)} bytes.")
            return video_bytes
        except Exception as e:
            logger.error(f"Error during T2V API generation for {model_key}: {e}", exc_info=True)
            error_msg = f"Failed to generate T2V video with {model_key}: {e}"
            raise ModelGenerationError(error_msg) from e

    def unload_model(self, model_key: str) -> None:
        """Unload a model from memory"""
        with self._model_lock:  # Global lock for model management operations
            if model_key in self._loaded_models:
                logger.info(f"Unloading model: {model_key}")
                del self._loaded_models[model_key]
                if self._device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

    def unload_all_models(self) -> None:
        """Unload all models from memory"""
        with self._model_lock:  # Global lock for model management operations
            logger.info("Unloading all models")
            self._loaded_models.clear()
            # --- Reset cache tracking on full unload --- #
            self._currently_loaded_model_key = None
            self._last_used_time = 0.0
            # --- End Reset --- #
            if self._device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

    # --- Video Preprocessing Helpers --- #
    def _aspect_ratio_resize(self, image, pipe, max_area=720 * 1280):
        """Resize image while maintaining aspect ratio, ensuring dimensions are multiples.
           Adapted from user provided example.
        """
        aspect_ratio = image.height / image.width
        # Calculate required modulo based on pipeline properties
        try:
            mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        except AttributeError:
            # Fallback for pipelines without these specific attributes (e.g., T2V?)
            # Using a common value like 16 or 8 might work
            mod_value = 16
            logger.warning(f"Could not determine optimal mod_value for {type(pipe).__name__}, falling back to {mod_value}")

        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        logger.debug(f"Resized source image to {width}x{height} (mod {mod_value})")
        return image, height, width

    def _center_crop_resize(self, image, target_height, target_width):
        """Center crop and resize image. Adapted from user provided example."""
        # Calculate resize ratio to match target dimensions
        resize_ratio = max(target_width / image.width, target_height / image.height)

        # Resize the image
        new_width = round(image.width * resize_ratio)
        new_height = round(image.height * resize_ratio)
        image = image.resize((new_width, new_height))

        # Center crop to target size
        image = TF.center_crop(image, [target_height, target_width]) # Note: TF expects [h, w]
        logger.debug(f"Center cropped image to {target_width}x{target_height}")
        return image
    # --- End Video Preprocessing Helpers --- #