"""
Image generation pipeline for CyberImage
"""
import threading
import queue
import time
import sys
import os
import gc
import torch
from typing import Dict, Optional
from PIL import Image
from flask import current_app
from app.models.manager import ModelManager
from app.utils.queue import QueueManager
from app.utils.image import ImageManager
from app.utils.watchdog import GPUWatchdog
from diffusers.utils import export_to_video
import uuid
from datetime import datetime
import numpy as np

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

class GenerationPipeline:
    """Handles the image generation process"""

    _instance = None  # Singleton instance
    _lock = threading.Lock()  # Lock for thread-safe singleton access
    _processing_lock = threading.Lock()  # Lock for ensuring single job processing
    _queue_event = threading.Event()  # Event to trigger queue processing

    # Add a dedicated generation lock to ensure only one generation happens at a time
    _generation_lock = threading.Lock()

    # Track if a generation is currently in progress
    _generation_in_progress = False
    _last_model_used = None

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GenerationPipeline, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        # Only initialize once
        if hasattr(self, 'is_initialized'):
            return

        # Check if we're in the main process
        self.is_main_process = (
            os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or  # Flask debug mode
            os.environ.get('GUNICORN_WORKER_ID') == '1'  # First Gunicorn worker
        )

        self.model_manager = None
        self.generation_queue = None
        self.is_running = False
        self.worker_thread = None
        self._app = current_app._get_current_object()
        self._processed_jobs = set()
        self.watchdog = None
        self.is_initialized = True

        # Only initialize pipeline in the main process
        if self.is_main_process:
            self._initialize()
        else:
            print("\n⚠️ Secondary process - generation disabled")
            sys.stdout.flush()

    def _initialize(self):
        """Initialize the pipeline components"""
        if self.is_running:
            return

        print("\n🚀 Initializing generation pipeline in main process...")
        sys.stdout.flush()

        # Reset any stuck jobs from previous runs
        try:
            # Reset any stalled processing jobs to pending
            reset_count = QueueManager.reset_stalled_jobs()
            if reset_count > 0:
                print(f"\n⚠️ Reset {reset_count} stalled jobs from previous run to pending status")
                sys.stdout.flush()

            # Check for pending jobs
            pending_jobs = QueueManager.get_pending_jobs()
            if pending_jobs:
                print(f"\n📋 Found {len(pending_jobs)} pending jobs")

            # Check for failed jobs that could be retried
            failed_jobs = QueueManager.get_failed_jobs()
            if failed_jobs:
                print(f"\n🔄 Found {len(failed_jobs)} failed jobs that can be retried")
                retried_count = 0

                for job in failed_jobs:
                    # Try to retry the job
                    job_id = job["id"]
                    print(f"   - Retrying failed job {job_id} (attempt #{job['retry_count'] + 1})")
                    retried_job_id = QueueManager.retry_failed_job(job_id)

                    if retried_job_id:
                        print(f"   - ✅ Job {job_id} reset to pending state")
                        retried_count += 1
                    else:
                        print(f"   - ❌ Could not retry job {job_id}")

                if retried_count > 0:
                    print(f"✅ Successfully retried {retried_count} failed jobs")
                    # Trigger the queue processing to pick up the retried jobs
                    self._queue_event.set()

            sys.stdout.flush()
        except Exception as e:
            print(f"\n❌ Error during queue recovery: {str(e)}")
            sys.stdout.flush()

        self.model_manager = ModelManager()
        self.generation_queue = queue.Queue()
        self.is_running = True

        # Initialize and start the watchdog
        self.watchdog = GPUWatchdog(
            model_manager=self.model_manager,
            app=self._app,
            max_stalled_time=3600,  # 15 minutes - increased to allow for longer generation times
            check_interval=60,     # Check every 30 seconds
            recovery_cooldown=300  # 5 minutes between recoveries
        )
        self.watchdog.start()

        # Start the background thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

        # Trigger initial queue check
        self._queue_event.set()

        print("✅ Generation pipeline initialized successfully")
        print("🔄 Queue processor started and waiting for jobs...")
        sys.stdout.flush()

    def ensure_running(self):
        """Ensure the pipeline is running, initializing if necessary"""
        if not self.is_running:
            self._initialize()

    def _force_memory_cleanup(self):
        """Force more aggressive memory cleanup"""
        print("\n🧹 Performing aggressive memory cleanup")
        sys.stdout.flush()

        try:
            # If we have a model manager, use it to unload all models
            if self.model_manager:
                if hasattr(self.model_manager, '_unload_all_models'):
                    self.model_manager._unload_all_models()
                elif hasattr(self.model_manager, 'unload_all_models'):
                    self.model_manager.unload_all_models()

            # Force CUDA cleanup if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()

                # Second pass cleanup
                torch.cuda.empty_cache()
                gc.collect()

                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                print(f"🧹 Memory after cleanup: {memory_allocated:.2f}GB")
                sys.stdout.flush()

            GenerationPipeline._generation_in_progress = False
            GenerationPipeline._last_model_used = None

        except Exception as e:
            print(f"⚠️ Error during memory cleanup: {str(e)}")
            sys.stdout.flush()

            # Reset state even if there was an error
            GenerationPipeline._generation_in_progress = False

    def _process_queue(self):
        """Background thread to process generation requests"""
        print("\n🔄 Queue processor thread started")
        sys.stdout.flush()

        with self._app.app_context():
            while self.is_running:
                try:
                    # Check if generation is in progress
                    if GenerationPipeline._generation_in_progress:
                        # Skip this iteration if there's already a generation in progress
                        time.sleep(1)
                        continue

                    # First check if any jobs are currently processing
                    queue_status = QueueManager.get_queue_status()
                    if queue_status["processing"] > 0:
                        # Skip this iteration if there's already a job processing
                        time.sleep(1)
                        continue

                    # Try to get the next job from the database
                    with self._processing_lock:
                        next_job = QueueManager.get_next_job()
                        if next_job is None:
                            # Wait for the queue event or timeout after 1 second
                            self._queue_event.wait(1)
                            self._queue_event.clear()
                            continue

                        # Process the job
                        print(f"\n🔄 Processing job {next_job['id']}")
                        sys.stdout.flush()
                        self.process_job(next_job)

                except Exception as e:
                    print(f"\n❌ Pipeline error: {str(e)}")
                    sys.stdout.flush()

                    # Reset generation flag if there was an error
                    GenerationPipeline._generation_in_progress = False
                    time.sleep(1)

    def process_job(self, job: Dict) -> Optional[str]:
        """Process a single generation job"""
        self.ensure_running()

        try:
            # --- Get Job Type --- #
            job_type = job.get("settings", {}).get("type", "image") # Default to image
            job_id = job["id"]
            model_id = job["model_id"]

            # Fetch model info for better logging
            model_info = current_app.config.get("AVAILABLE_MODELS", {}).get(model_id, {})
            model_description = model_info.get("description", "Unknown model")
            model_type = model_info.get("type", "unknown")

            # Update job status to processing
            QueueManager.update_job_status(job_id, "processing", "Initializing...")

            # Enhanced Job Logging
            border = "=" * 50
            print(f"\n{border}")
            print(f"📝 PROCESSING JOB: {job_id}")
            print(f"📝 MODEL: {model_id} - {model_description} (Type: {model_type})")
            print(f"📝 JOB TYPE: {job_type.upper()}")

            if "prompt" in job:
                print(f"📝 PROMPT: {job['prompt'][:100]}" + ('...' if len(job['prompt']) > 100 else ''))
            if "negative_prompt" in job and job["negative_prompt"]:
                print(f"📝 NEGATIVE: {job['negative_prompt'][:100]}" + ('...' if len(job['negative_prompt']) > 100 else ''))

            # Job specific details
            if job_type == "i2v" and "source_image_id" in job.get("settings", {}):
                print(f"📝 SOURCE IMAGE: {job['settings']['source_image_id']}")

            print(f"{border}")
            sys.stdout.flush()
            # End Enhanced Job Logging

            # --- Handle Retries --- #
            metadata = job.get("settings", {})
            retry_count = metadata.get("retry_count", 0)
            original_job_id = metadata.get("original_job_id")

            if retry_count > 0:
                print(f"   • Retry #{retry_count}" +
                      (f" of job {original_job_id}" if original_job_id else ""))

            # --- Filter Settings --- #
            # Filter out admin-specific flags and job type that shouldn't be passed to the model
            core_settings = {k: v for k, v in job.get("settings", {}).items()
                             if k not in ["type", "admin_retried", "admin_retry_timestamp", "retry_count", "num_images", "source_image_id"]}

            # --- Acquire Generation Lock --- #
            with self._generation_lock:
                generated_media_ids = []
                try:
                    # Mark that generation is in progress
                    GenerationPipeline._generation_in_progress = True
                    GenerationPipeline._last_model_used = job["model_id"]

                    # --- Branch based on Job Type --- #

                    # --- Text-to-Video (T2V) Generation --- #
                    if job_type == "t2v":
                        print(f"🎬 Starting Text-to-Video generation (T2V)")
                        sys.stdout.flush()
                        QueueManager.update_job_status(job_id, "processing", "Loading T2V model...")

                        # Generate video frames
                        QueueManager.update_job_status(job_id, "processing", "Generating T2V frames...")
                        frames = self.model_manager.generate_text_to_video(
                            job["model_id"],
                            job["prompt"],
                            **core_settings
                        )

                        # Check if frames generation failed or returned empty (Simplified Check)
                        if frames is None or not hasattr(frames, '__len__') or len(frames) == 0:
                            raise ValueError("T2V generation returned no valid frames")

                        # Save video
                        QueueManager.update_job_status(job_id, "processing", "Saving T2V video...")
                        fps = job["settings"].get("fps", 16)

                        # --- Generate Path for Video --- #
                        video_id = str(uuid.uuid4())
                        today = datetime.utcnow().strftime("%Y/%m/%d")
                        video_dir = os.path.join(current_app.config["IMAGES_PATH"], today)
                        os.makedirs(video_dir, exist_ok=True)
                        file_name = f"{video_id}.mp4"
                        relative_video_path = os.path.join(today, file_name)
                        output_video_path = os.path.join(video_dir, file_name)
                        # --- End Path Generation --- #

                        # --- DEBUG: Inspect frames before export (REMOVED CONVERSION) ---
                        print(f"DEBUG: Before export_to_video - frames type: {type(frames)}")
                        if isinstance(frames, np.ndarray):
                            print(f"DEBUG: Frames are ndarray. Shape: {frames.shape}, Dtype: {frames.dtype}")
                        elif isinstance(frames, list) and len(frames) > 0:
                            print(f"DEBUG: Frames are list. Length: {len(frames)}, First element type: {type(frames[0])}")
                        elif torch.is_tensor(frames):
                            print(f"DEBUG: Frames are tensor. Shape: {frames.shape}, Dtype: {frames.dtype}")
                        sys.stdout.flush()
                        # --- End DEBUG ---

                        export_to_video(frames, output_video_path, fps=fps)

                        # Prepare metadata for video
                        duration_seconds = len(frames) / fps
                        video_metadata = {
                            "model_id": job["model_id"],
                            "prompt": job["prompt"],
                            "settings": job["settings"], # Save all settings including type
                            "generation_time": time.time(),
                            "fps": fps,
                            "duration_seconds": duration_seconds,
                            "frame_count": len(frames),
                            "type": "video" # Use generic 'video' type for metadata
                        }

                        # Save video record using the updated save_image method
                        ImageManager.save_image(None, job_id, video_metadata, image_id=video_id, file_path=relative_video_path)
                        generated_media_ids.append(video_id)
                        print(f"✅ Saved T2V video: {video_id} with {len(frames)} frames")
                        sys.stdout.flush()

                    # --- Image-to-Video (I2V) Generation --- #
                    elif job_type == "i2v": # Use the specific type from config/API
                        print(f"🎬 Starting Image-to-Video generation (I2V)")
                        sys.stdout.flush()

                        source_image_id = job["settings"].get("source_image_id")
                        if not source_image_id:
                            raise ValueError("Missing source_image_id for I2V job")

                        QueueManager.update_job_status(job_id, "processing", "Loading I2V model...")

                        # Get source image path
                        source_image_path = ImageManager.get_image_path(source_image_id)
                        if not source_image_path:
                            raise ValueError(f"I2V source image path not found for ID: {source_image_id}")

                        # Generate video frames from image
                        QueueManager.update_job_status(job_id, "processing", "Generating I2V frames...")
                        frames = self.model_manager.generate_image_to_video(
                            job["model_id"],
                            job["prompt"], # This is the video prompt for I2V
                            source_image_path,
                            **core_settings # Pass filtered settings
                        )

                        # Check if frames generation failed or returned empty (Simplified Check)
                        if frames is None or not hasattr(frames, '__len__') or len(frames) == 0:
                            raise ValueError("I2V generation returned no valid frames")

                        # Save video
                        QueueManager.update_job_status(job_id, "processing", "Saving I2V video...")
                        fps = job["settings"].get("fps", 16)

                        # --- Generate Path for Video --- #
                        video_id = str(uuid.uuid4())
                        today = datetime.utcnow().strftime("%Y/%m/%d")
                        video_dir = os.path.join(current_app.config["IMAGES_PATH"], today)
                        os.makedirs(video_dir, exist_ok=True)
                        file_name = f"{video_id}.mp4"
                        relative_video_path = os.path.join(today, file_name)
                        output_video_path = os.path.join(video_dir, file_name)
                        # --- End Path Generation --- #

                        # --- DEBUG: Inspect frames before export (REMOVED CONVERSION) ---
                        print(f"DEBUG: Before export_to_video (I2V) - frames type: {type(frames)}")
                        if isinstance(frames, np.ndarray):
                            print(f"DEBUG: Frames are ndarray (I2V). Shape: {frames.shape}, Dtype: {frames.dtype}")
                        elif isinstance(frames, list) and len(frames) > 0:
                            print(f"DEBUG: Frames are list (I2V). Length: {len(frames)}, First element type: {type(frames[0])}")
                        elif torch.is_tensor(frames):
                             print(f"DEBUG: Frames are tensor (I2V). Shape: {frames.shape}, Dtype: {frames.dtype}")
                        sys.stdout.flush()
                        # --- End DEBUG ---

                        export_to_video(frames, output_video_path, fps=fps)

                        # Prepare metadata for video
                        duration_seconds = len(frames) / fps
                        video_metadata = {
                            "model_id": job["model_id"],
                            "prompt": job["prompt"],
                            "settings": job["settings"], # Save all settings including type and source_id
                            "generation_time": time.time(),
                            "fps": fps,
                            "duration_seconds": duration_seconds,
                            "frame_count": len(frames),
                            "type": "video" # Use generic 'video' type for metadata
                        }

                        # Save video record using the updated save_image method
                        ImageManager.save_image(None, job_id, video_metadata, image_id=video_id, file_path=relative_video_path)
                        generated_media_ids.append(video_id)
                        print(f"✅ Saved I2V video: {video_id} with {len(frames)} frames")
                        sys.stdout.flush()

                    # --- Image Generation (Default/Existing) --- #
                    # This implicitly handles job_type == "image" or if type is missing
                    else:
                        print(f"🖼️ Starting Image generation")
                        sys.stdout.flush()

                        num_images = job["settings"].get("num_images", 1)

                        for i in range(num_images):
                            print(f"   ▷ Generating image {i+1}/{num_images} (Job: {job_id})")
                            sys.stdout.flush()

                            QueueManager.update_job_status(job_id, "processing", f"Loading model for image {i+1}/{num_images}...")

                            # Generate the image
                            QueueManager.update_job_status(job_id, "processing", f"Generating image {i+1}/{num_images}...")
                            image = self.model_manager.generate_image(
                                job["model_id"],
                                job["prompt"],
                                negative_prompt=job.get("negative_prompt"),
                                **core_settings
                            )

                            # Prepare metadata for the image
                            image_metadata = {
                                "model_id": job["model_id"],
                                "prompt": job["prompt"],
                                "negative_prompt": job.get("negative_prompt"),
                                "settings": job["settings"], # Save all original settings
                                "generation_time": time.time(),
                                "image_number": i + 1,
                                "total_images": num_images,
                                "type": "image" # Explicitly store type
                            }

                            # Save the image with metadata and job ID
                            image_id = ImageManager.save_image(image, job_id, image_metadata)
                            generated_media_ids.append(image_id)
                            print(f"   ✅ Saved image: {image_id} (Image {i+1}/{num_images})")
                            sys.stdout.flush()

                finally:
                    # Always mark generation as complete
                    GenerationPipeline._generation_in_progress = False
                    # Keep model potentially cached
                    pass

            # Update job status to completed
            # Store the list of generated image/video IDs in the job record
            # Determine the correct media type string for the completion message
            final_media_type = "image" # Default
            if job_type == "t2v" or job_type == "i2v":
                 final_media_type = "video"

            completion_message = f"Generated {len(generated_media_ids)} {final_media_type}(s)"
            QueueManager.update_job_status(
                job_id,
                "completed",
                completion_message,
                image_ids=generated_media_ids # Changed from media_ids to image_ids to match API
            )

            print(f"✅ JOB COMPLETED: {job_id} - Generated {len(generated_media_ids)} {final_media_type}(s)")
            sys.stdout.flush()

            # Return the first generated ID
            return generated_media_ids[0] if generated_media_ids else None

        except Exception as e:
            # Handle generation errors
            job_id = job.get("id", "unknown")
            exception_type = type(e).__name__
            error_message = f"Error processing job {job_id} ({job_type}): {str(e)}"

            # --- Enhanced Error Logging --- #
            print(f"\n❌ {error_message}")
            print(f"   • Exception Type: {exception_type}")
            # Ensure message is flushed *before* potential exit
            sys.stdout.flush()
            # Also log to logger for persistence
            current_app.logger.error(f"Failed job {job_id}: {error_message}", exc_info=True)
            # --- End Enhanced Logging --- #

            # Update job status to failed
            QueueManager.update_job_status(job_id, "failed", error_message)

            # Make sure generation is marked as not in progress
            GenerationPipeline._generation_in_progress = False

            # Force memory cleanup on error
            self._force_memory_cleanup()

            # Terminate the application if it's a critical model/generation error
            # Avoid terminating for recoverable errors like missing source image
            if isinstance(e, (ValueError, FileNotFoundError)):
                 print(f"   • Recoverable error, continuing queue processing.")
                 sys.stdout.flush() # Flush this message too
            else:
                # --- Log Termination --- #
                termination_message = f"\n🚨 Critical failure ({exception_type}) in generation for job {job_id}. Terminating process."
                print(termination_message)
                current_app.logger.critical(termination_message)
                sys.stdout.flush()
                # --- End Log Termination --- #
                os._exit(1) # Immediate termination

            return None

    def add_job(self, job: Dict):
        """Add a job to the generation queue"""
        self.ensure_running()

        print(f"\n📥 Adding job {job.get('id', 'unknown')} to queue processing")
        sys.stdout.flush()

        # Validate that we have the necessary information
        if 'id' not in job:
            raise ValueError("Job must have an ID")

        # The job has already been added to the database by the API route
        # We don't need to add it again, just trigger the queue processing
        # to pick up the job from the database

        # Notify the queue processor to check for new jobs
        self._queue_event.set()

        # Return the job ID
        return job.get('id')

    def stop(self):
        """Stop the generator pipeline"""
        if not self.is_running:
            return

        print("\n🛑 Stopping generation pipeline...")
        sys.stdout.flush()

        # Stop the watchdog first
        if self.watchdog:
            self.watchdog.stop()
            print("✅ Watchdog stopped")

        # Stop the queue processor
        self.is_running = False
        self._queue_event.set()  # Wake up the queue processor to stop

        # Force cleanup of any loaded models
        self._force_memory_cleanup()

        # Give the thread time to complete
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
            print("✅ Worker thread stopped")

        # Clean up any remaining resources
        self.model_manager = None
        self.generation_queue = None
        print("✅ Generation pipeline stopped")
        sys.stdout.flush()