"""
Model management system for CyberImage
"""
import logging
import sys
import platform
import gc
import os
from typing import Dict, Optional, Union
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
from diffusers import LTXPipeline, LTXVideoTransformer3DModel
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
        # Generation is now handled by the GenerationPipeline class
        # This class only manages loading and unloading models

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

    def get_model(self, model_key: str) -> Optional[Union[FluxPipeline, DiffusionPipeline]]:
        """Get a model by its key, loading it if necessary"""
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        with self._model_lock:
            current_time = time.time()

            # --- Enhanced Logging --- #
            model_info = AVAILABLE_MODELS.get(model_key, {})
            model_desc = model_info.get("description", "Unknown model")
            model_type = model_info.get("type", "Unknown type")
            print(f"\n🧠 MODEL REQUEST: {model_key} - {model_desc} (Type: {model_type})")
            sys.stdout.flush()
            # --- End Enhanced Logging --- #

            # --- Check Cache ---
            if (model_key == self._currently_loaded_model_key and
                model_key in self._loaded_models and
                (current_time - self._last_used_time < self._cache_duration)):

                try:
                    # Verify model is still valid (basic check)
                    model = self._loaded_models[model_key]
                    if hasattr(model, 'device'):
                        print(f"✅ Using cached model: {model_key} ({model_desc}) - was loaded {current_time - self._last_used_time:.1f}s ago")
                        sys.stdout.flush()
                        logger.info(f"Using cached model: {model_key} (last used {current_time - self._last_used_time:.1f}s ago)")
                        self._last_used_time = current_time # Update last used time on access
                        return model
                    else:
                        logger.warning(f"Cached model {model_key} seems invalid (no device attribute), reloading...")
                except Exception as e:
                    logger.warning(f"Error accessing cached model {model_key}, reloading... Error: {str(e)}")
            # --- End Cache Check ---

            # --- Load or Reload Model ---
            # If cache miss, different model, or timeout expired, unload existing and load new
            if self._currently_loaded_model_key:
                print(f"🔄 Unloading previous model ({self._currently_loaded_model_key}) to load {model_key} ({model_desc})")
                sys.stdout.flush()
                logger.info(f"Unloading previous model ({self._currently_loaded_model_key}) to load {model_key}")
            else:
                print(f"🔄 Loading model: {model_key} ({model_desc}) - Type: {model_type}")
                sys.stdout.flush()
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
                    # Determine dtype for video model (needs careful consideration)
                    # Using float32 for VAE/Image Encoder as per example, bfloat16 for pipeline
                    # This might need adjustment based on hardware/performance testing
                    video_dtype = torch.bfloat16 if self._dtype == torch.bfloat16 else torch.float16
                    vae_dtype = torch.float32
                    encoder_dtype = torch.float32

                    logger.debug(f"Loading video components: VAE ({vae_dtype}), Image Encoder ({encoder_dtype}), Pipeline ({video_dtype})")

                    image_encoder = CLIPVisionModel.from_pretrained(
                        model_config['repo'],
                        subfolder="image_encoder",
                        torch_dtype=encoder_dtype,
                        local_files_only=True, # Assume local for now
                        cache_dir=model_path
                    )
                    vae = AutoencoderKLWan.from_pretrained(
                        model_path,
                        subfolder="vae",
                        torch_dtype=vae_dtype,
                        local_files_only=True, # Assume local for now
                        cache_dir=model_path
                    )
                    pipe = WanImageToVideoPipeline.from_pretrained(
                        model_config['repo'],
                        vae=vae,
                        image_encoder=image_encoder,
                        torch_dtype=video_dtype,
                        local_files_only=True, # Assume local for now
                        cache_dir=model_path
                    )

                    # --- Conditional Memory Optimization for Video Models --- #
                    # Check config for explicit offload setting, default to True if not specified
                    should_offload = model_config.get("step_config", {}).get("offload", True)
                    # Example: Don't offload specific small models like wan-t2v-1.3b
                    if model_key == "wan-t2v-1.3b":
                        should_offload = False

                    if should_offload:
                        try:
                            pipe = apply_memory_optimizations(pipe)
                            logger.info(f"Applied memory optimizations (CPU offload) to video model {model_key}")
                        except Exception as mem_e:
                            logger.warning(f"Could not apply standard CPU offload to {model_key}: {mem_e}. Moving to device directly.")
                            pipe.to(self._device)
                    else:
                        logger.info(f"Skipping CPU offload for video model {model_key}, loading directly to {self._device}")
                        pipe.to(self._device)
                    # --- End Conditional Optimization --- #

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

    def generate_image_to_video(self, model_key: str, video_prompt: str, source_image_path: str, **kwargs) -> list:
        """Generate video frames using an Image-to-Video model and a source image."""
        try:
            # Enhanced logging
            model_info = AVAILABLE_MODELS.get(model_key, {})
            model_desc = model_info.get("description", "Unknown model")
            print(f"\n🎬 GENERATING I2V VIDEO with model: {model_key} - {model_desc}")
            print(f"   • Source Image: {source_image_path}")
            print(f"   • Video Prompt: {video_prompt[:100]}" + ('...' if len(video_prompt) > 100 else ''))

            # Log key parameters
            guidance = kwargs.get("guidance_scale", 5.5)
            fps = kwargs.get("fps", 16)
            print(f"   • Parameters: Guidance={guidance}, FPS={fps}")
            sys.stdout.flush()
            # End enhanced logging

            # 1. Get model
            pipe = self.get_model(model_key)
            if pipe is None:
                raise ModelLoadError(f"Failed to get video model {model_key}")

            # Check if it's the correct pipeline type (optional but good practice)
            if not isinstance(pipe, WanImageToVideoPipeline):
                logger.warning(f"Model {model_key} loaded, but is not a WanImageToVideoPipeline. Type: {type(pipe).__name__}")
                # Attempt to proceed anyway, or raise error?
                # raise TypeError(f"Model {model_key} is not the expected video pipeline type.")

            # 2. Load source image
            source_image = load_image(source_image_path)

            # 3. Apply preprocessing
            # Resize first frame based on aspect ratio and model constraints
            # Use a default max area, or allow override via kwargs?
            max_area = kwargs.get("max_video_area", 720 * 1280)
            processed_image, height, width = self._aspect_ratio_resize(source_image, pipe, max_area=max_area)

            # Note: The example code used a separate `last_frame`. For simplicity,
            # we are currently only using the single source_image.
            # If last_frame functionality is needed later, it would be loaded and processed here.
            # e.g., last_frame = self._center_crop_resize(load_image(last_frame_path), height, width)

            # 4. Prepare arguments
            pipe_args = {
                "image": processed_image,
                # "last_image": last_frame, # If using last frame
                "prompt": video_prompt,
                "height": height,
                "width": width,
                "guidance_scale": kwargs.get("guidance_scale", 5.5) # Use provided or default
                # Add other relevant video pipeline args from kwargs if needed
                # e.g., "num_inference_steps", "num_frames"
            }

            # Filter out any kwargs not expected by the pipeline (optional)
            # expected_args = set(inspect.signature(pipe.__call__).parameters.keys())
            # filtered_args = {k: v for k, v in pipe_args.items() if k in expected_args}

            # Log generation start
            print(f"\n🚀 Starting I2V generation with {model_key} (size: {width}x{height})")
            sys.stdout.flush()

            # 5. Call the pipeline
            # Assuming the pipeline returns frames in `.frames[0]` based on example
            output = pipe(**pipe_args).frames[0]

            print(f"✅ Successfully generated {len(output)} frames with model {model_key}")
            sys.stdout.flush()
            logger.info(f"Successfully generated {len(output)} frames with model {model_key}")

            # 6. Return frames (as PIL Images)
            return output

        except Exception as e:
            error_msg = f"Failed to generate I2V video with {model_key}: {str(e)}"
            logger.error(error_msg, exc_info=True) # Log traceback
            # Force cleanup on error
            self._force_memory_cleanup()
            raise ModelGenerationError(error_msg)

    def generate_text_to_video(self, model_key: str, prompt: str, **kwargs) -> list:
        """Generate video frames using a Text-to-Video model."""
        try:
            # Enhanced logging
            model_info = AVAILABLE_MODELS.get(model_key, {})
            model_desc = model_info.get("description", "Unknown model")
            model_type = model_info.get("type", "unknown") # Get model type for logic
            print(f"\n🎬 GENERATING T2V VIDEO with model: {model_key} - {model_desc}")
            print(f"   • Text Prompt: {prompt[:100]}" + ('... ' if len(prompt) > 100 else ''))

            # Log key parameters from kwargs or defaults
            guidance = kwargs.get("guidance_scale", 7.0)
            steps = kwargs.get("num_inference_steps", 50)
            num_frames = kwargs.get("num_frames", 161) # Default from LTX example
            width = kwargs.get("width", 704) # Default from LTX example
            height = kwargs.get("height", 480) # Default from LTX example
            # --- Use default negative prompt if none provided --- #
            provided_negative_prompt = kwargs.get("negative_prompt", "")
            default_negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
            negative_prompt = provided_negative_prompt if provided_negative_prompt else default_negative_prompt
            # --- End default negative prompt --- #

            print(f"   • Parameters: Steps={steps}, Guidance={guidance}, Frames={num_frames}, Size={width}x{height}")
            if negative_prompt:
                 print(f"   • Negative Prompt: {negative_prompt[:100]}" + ('... ' if len(negative_prompt) > 100 else ''))
            sys.stdout.flush()
            # End enhanced logging

            # 1. Get model (could be LTXPipeline, WanPipeline, etc.)
            pipe = self.get_model(model_key)
            if pipe is None:
                raise ModelLoadError(f"Failed to get T2V model {model_key}")

            # 2. Prepare arguments based on pipeline type
            pipe_args = {
                "prompt": prompt,
                "guidance_scale": guidance,
                "num_inference_steps": steps,
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "negative_prompt": negative_prompt # Use the potentially defaulted negative_prompt
            }

            # Log generation start
            print(f"\n🚀 Starting T2V generation with {model_key} ({pipe_args['num_inference_steps']} steps)")
            sys.stdout.flush()

            # 3. Call the pipeline
            result = pipe(**pipe_args)

            # --- DEBUG: Log result type and attributes ---
            logger.debug(f"T2V pipe result type: {type(result)}")
            if hasattr(result, '__dict__'):
                logger.debug(f"T2V pipe result attributes: {result.__dict__.keys()}")
            elif hasattr(result, 'keys'):
                 logger.debug(f"T2V pipe result keys: {result.keys()}")

            if hasattr(result, 'frames'):
                 logger.debug(f"T2V pipe result.frames type: {type(result.frames)}")
                 if isinstance(result.frames, (list, np.ndarray)) and hasattr(result.frames, '__len__'):
                     logger.debug(f"T2V pipe result.frames length: {len(result.frames)}")
                     if len(result.frames) > 0:
                        logger.debug(f"T2V pipe result.frames[0] type: {type(result.frames[0])}")
                        if hasattr(result.frames[0], 'shape'):
                             logger.debug(f"T2V pipe result.frames[0] shape: {result.frames[0].shape}")
            # --- END DEBUG ---

            # 4. Extract frames based on expected output structure (Aligned with LTX reference)
            output_frames = None
            if hasattr(result, 'frames'):
                # Reference code uses result.frames[0] directly
                # Check if it's a non-empty list/array first
                if isinstance(result.frames, (list, tuple, np.ndarray)) and len(result.frames) > 0:
                    output_frames = result.frames[0]
                    logger.info(f"Extracted frames from result.frames[0] (type: {type(output_frames)})")
                elif result.frames is not None:
                     # Handle cases where result.frames might be the frames directly
                     output_frames = result.frames
                     logger.warning(f"Using result.frames directly as output (type: {type(output_frames)}). Structure might differ from expected result.frames[0].")
                else:
                    logger.error("T2V result has 'frames' attribute, but it is empty or None.")
            elif hasattr(result, 'images'): # Fallback for older/different pipelines
                output_frames = result.images
                logger.warning(f"T2V result missing 'frames', falling back to 'images' attribute.")
            else:
                 logger.error(f"T2V pipeline {model_key} result object missing 'frames' and 'images' attributes.")
                 raise ModelGenerationError(f"T2V pipeline {model_key} did not return expected output structure.")

            # Check if extraction failed
            if output_frames is None:
                logger.error(f"Failed to extract frames from T2V pipeline result. Result object: {result}")
                raise ModelGenerationError(f"Failed to extract valid frames from T2V pipeline {model_key} result.")

            # --- Added DEBUG: Inspect output_frames before returning --- #
            if output_frames is not None and hasattr(output_frames, '__len__') and len(output_frames) > 0:
                first_frame = output_frames[0]
                logger.debug(f"T2V generate_text_to_video: output_frames type: {type(output_frames)}")
                logger.debug(f"T2V generate_text_to_video: first_frame type: {type(first_frame)}")
                if isinstance(first_frame, torch.Tensor):
                    logger.debug(f"T2V generate_text_to_video: first_frame tensor dtype: {first_frame.dtype}")
                    logger.debug(f"T2V generate_text_to_video: first_frame tensor shape: {first_frame.shape}")
                    # Log min/max ONLY if it's a float tensor to avoid issues with other types
                    if first_frame.dtype.is_floating_point:
                        logger.debug(f"T2V generate_text_to_video: first_frame tensor min/max: {first_frame.min()}, {first_frame.max()}")
                elif isinstance(first_frame, Image.Image):
                    logger.debug(f"T2V generate_text_to_video: first_frame PIL mode: {first_frame.mode}")
                    logger.debug(f"T2V generate_text_to_video: first_frame PIL size: {first_frame.size}")
            else:
                logger.debug("T2V generate_text_to_video: output_frames is None or empty.")
            # --- End DEBUG --- #

            # Get number of frames correctly (more robust check)
            num_generated_frames = 0
            try:
                # Try standard len first
                num_generated_frames = len(output_frames)
            except TypeError:
                # Handle cases like single tensors that might not have len()
                if hasattr(output_frames, 'shape'):
                    num_generated_frames = output_frames.shape[0] # Assume first dimension is frames
                else:
                    logger.warning(f"Could not determine length/frame count from output_frames (type: {type(output_frames)}). Assuming 1 frame.")
                    num_generated_frames = 1 # Or handle as error if length is critical

            print(f"✅ Successfully generated {num_generated_frames} T2V frames with model {model_key}")
            sys.stdout.flush()
            logger.info(f"Successfully generated {num_generated_frames} T2V frames with model {model_key}")
            return output_frames # Return the frames (NumPy array or List)

        except Exception as e:
            error_msg = f"Failed to generate T2V video with {model_key}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._force_memory_cleanup()
            raise ModelGenerationError(error_msg)

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