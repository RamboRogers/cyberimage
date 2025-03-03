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
from diffusers import FluxPipeline, DiffusionPipeline, StableDiffusionPipeline, BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
from app.models import AVAILABLE_MODELS
from tqdm import tqdm
from threading import Lock
from flask import current_app
import time
import threading

# Configure logging
logger = logging.getLogger(__name__)

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
        logger.info(f"ModelManager initialized (device={self._device}, model_cache_size={self._max_models})")

        # Add last health check timestamp
        self._last_health_check = time.time()
        self._health_check_interval = 60  # Check every 60 seconds

    def check_gpu_health(self):
        """Check GPU memory usage and force cleanup if necessary"""
        if self._device != "cuda":
            return True  # Only applicable for CUDA devices

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
            if memory_allocated / total_memory > 0.8:
                logger.info(f"GPU Health Check: {memory_allocated:.2f}GB/{total_memory:.2f}GB ({memory_allocated/total_memory:.2%})")

            # Check if we're approaching a dangerous threshold (e.g., 90% usage)
            memory_threshold = total_memory * 0.9
            if memory_allocated > memory_threshold:
                logger.warning(f"GPU memory usage exceeds threshold ({memory_allocated:.2f}GB > {memory_threshold:.2f}GB)")
                self._force_memory_cleanup()
                return False  # Indicate unhealthy state

            return True  # Healthy
        except Exception as e:
            logger.error(f"Error checking GPU health: {str(e)}")
            return False  # Consider unhealthy on error

    def _force_memory_cleanup(self):
        """Force aggressive memory cleanup"""
        logger.info("Forcing complete GPU memory cleanup")

        try:
            # Unload all models
            self._unload_all_models()

            # Force CUDA cache flush
            if self._device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()  # Wait for all CUDA operations to complete

                # Second pass cleanup
                torch.cuda.empty_cache()
                gc.collect()

                # Print memory state after cleanup
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"GPU memory after cleanup: {memory_allocated:.2f}GB")
        except Exception as e:
            logger.error(f"Error during forced cleanup: {str(e)}")

    def load_sd_pipeline(self, model_key: str) -> StableDiffusion3Pipeline:
        """Load SD 3.5 with proper configuration"""
        model_id = AVAILABLE_MODELS[model_key]["id"]

        logger.info(f"Loading SD 3.5 model")

        # Configure 4-bit quantization
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load transformer with quantization
        transformer = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.float16,
            cache_dir="models/sd-3.5"
        )

        # Load full pipeline
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=transformer,
            torch_dtype=torch.float16,
            cache_dir="models/sd-3.5"
        )

        # Enable CPU offloading as primary memory optimization
        pipeline.enable_model_cpu_offload()
        logger.debug("Enabled CPU offloading for SD 3.5 model")

        return pipeline

    def get_model(self, model_key: str) -> Optional[Union[FluxPipeline, DiffusionPipeline]]:
        """Get a model by its key, loading it if necessary"""
        if model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_key}")

        with self._model_lock:
            # Check if model is loaded and valid
            if model_key in self._loaded_models:
                try:
                    model = self._loaded_models[model_key]
                    # Verify model is in a valid state
                    if hasattr(model, 'device'):
                        logger.debug(f"Using already loaded model: {model_key}")
                        return model
                except Exception as e:
                    logger.warning(f"Cached model {model_key} is invalid, reloading... Error: {str(e)}")
                    del self._loaded_models[model_key]

            # ALWAYS unload all models and clear GPU memory before loading a new one
            logger.info(f"Loading model: {model_key} (forcing cleanup first)")
            self._force_memory_cleanup()

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

                # Common configuration for all models
                model_config = {
                    "torch_dtype": torch.float16,  # Use float16 for better memory efficiency
                    "local_files_only": True,
                }

                # For CUDA devices, we'll rely on CPU offloading rather than device_map
                logger.debug("Using CPU offloading strategy for memory management")

                # Common memory optimizations for all models
                def apply_memory_optimizations(pipe):
                    # Use CPU offloading as the primary memory optimization strategy
                    # This will automatically handle moving only necessary components to GPU during inference
                    pipe.enable_model_cpu_offload()
                    logger.debug("Enabled CPU offloading to minimize GPU memory usage")

                    # Print memory usage after optimizations
                    if self._device == "cuda":
                        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                        logger.debug(f"GPU memory after optimizations: {memory_allocated:.2f}GB")

                    return pipe

                if model_key == "flux-1":
                    pipe = FluxPipeline.from_pretrained(
                        model_path,
                        **model_config
                    )
                    pipe = apply_memory_optimizations(pipe)
                    logger.info(f"Loaded FLUX-1 model with CPU offloading")

                elif model_key == "sd-3.5":
                    # SD 3.5 has special handling with transformer
                    pipe = self.load_sd_pipeline(model_key)

                elif model_key == "flux-abliterated":
                    pipe = FluxPipeline.from_pretrained(
                        model_path,
                        **model_config
                    )
                    pipe = apply_memory_optimizations(pipe)
                    logger.info(f"Loaded FLUX-Abliterated model with CPU offloading")

                if pipe is None:
                    raise ValueError(f"Failed to load model {model_key}")

                # Verify the model is on the correct device
                if hasattr(pipe, 'device'):
                    logger.debug(f"Model device: {pipe.device}")
                    # We're using CPU offloading, so we don't want to manually move to GPU
                    # if str(pipe.device) != self._device and self._device != "mps":
                    #     print(f"   • Moving model from {pipe.device} to {self._device}")
                    #     pipe = pipe.to(self._device)

                # Update memory tracking after loading
                if self._device == "cuda":
                    torch.cuda.synchronize()
                    memory_allocated = torch.cuda.memory_allocated() / (1024**2)
                    memory_reserved = torch.cuda.memory_reserved() / (1024**2)
                    logger.debug(f"Post-load GPU Memory: {memory_allocated:.2f}MB, Reserved: {memory_reserved:.2f}MB")
                    torch.cuda.empty_cache()
                    gc.collect()

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

        if self._device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            logger.debug(f"GPU memory after cleanup: {memory_allocated:.2f}GB")

    def generate_image(self, model_key: str, prompt: str, **kwargs) -> torch.Tensor:
        """Generate an image using the specified model"""
        try:
            # Get the model
            pipe = self.get_model(model_key)
            if pipe is None:
                raise ModelLoadError(f"Failed to get model {model_key}")

            # Set default generation parameters if not provided
            kwargs.setdefault("num_inference_steps", 30)
            kwargs.setdefault("guidance_scale", 7.5)
            kwargs.setdefault("height", 1024)
            kwargs.setdefault("width", 1024)
            kwargs.setdefault("max_sequence_length", 512)

            # Handle negative prompt based on model type
            negative_prompt = kwargs.pop("negative_prompt", None)
            if negative_prompt and model_key == "sd-3.5":
                kwargs["negative_prompt"] = negative_prompt

            # Remove callback parameters - neither Flux nor SD-3.5 implementations seem to support them
            kwargs.pop("callback", None)
            kwargs.pop("callback_steps", None)

            logger.info(f"Generating image with {model_key}: {prompt[:50]}...")

            # Create generator on appropriate device
            if self._device == "mps":
                generator = torch.Generator("cpu").manual_seed(
                    kwargs.get("seed", None) or torch.randint(1, 9999999, (1,)).item()
                )
            else:
                generator = torch.Generator(device=self._device).manual_seed(
                    kwargs.get("seed", None) or torch.randint(1, 9999999, (1,)).item()
                )

            # Standard generation for all models - no callbacks
            result = pipe(
                prompt,
                generator=generator,
                **kwargs
            ).images[0]

            # Success!
            logger.info(f"Successfully generated image with model {model_key}")
            return result

        except Exception as e:
            error_msg = f"Failed to generate image: {str(e)}"
            logger.error(error_msg)
            # Force cleanup on error
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
            if self._device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()