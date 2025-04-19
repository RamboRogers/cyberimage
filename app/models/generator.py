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
            max_stalled_time=900,  # 15 minutes - increased to allow for longer generation times
            check_interval=30,     # Check every 30 seconds
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
            # Update job status to processing
            QueueManager.update_job_status(job["id"], "processing", "Initializing model...")
            print(f"\n🎨 Starting image generation for job {job['id']}")
            sys.stdout.flush()

            # Extract metadata if available (for retry tracking)
            metadata = job.get("metadata", {})
            retry_count = metadata.get("retry_count", 0)
            original_job_id = metadata.get("original_job_id")

            if retry_count > 0:
                print(f"   • Retry #{retry_count}" +
                      (f" of job {original_job_id}" if original_job_id else ""))

            # Get number of images to generate
            num_images = job["settings"].get("num_images", 1)
            image_ids = []

            # Create settings without num_images to pass to generate_image
            generation_settings = {k: v for k, v in job["settings"].items() if k != "num_images"}

            # Filter out admin-specific flags that shouldn't be passed to the model
            admin_flags = ["admin_retried", "admin_retry_timestamp", "retry_count"]
            generation_settings = {k: v for k, v in generation_settings.items() if k not in admin_flags}

            # Get the total inference steps for progress tracking
            total_steps = generation_settings.get("num_inference_steps", 30)

            # Acquire the generation lock - this ensures only one generation happens at a time
            with self._generation_lock:
                try:
                    # Mark that generation is in progress to prevent other threads from starting generation
                    GenerationPipeline._generation_in_progress = True

                    # Force cleanup before we start, especially if we've had previous generations
                    # --- Comment out initial cleanup to allow caching ---
                    # self._force_memory_cleanup()
                    # --- --- --- --- --- --- --- --- --- --- --- --- ---

                    # Record which model we're about to use
                    GenerationPipeline._last_model_used = job["model_id"]

                    for i in range(num_images):
                        # Update status for generation
                        QueueManager.update_job_status(
                            job["id"],
                            "processing",
                            f"Generating image {i+1} of {num_images}..."
                        )

                        # Load model and update status for each image individually
                        QueueManager.update_job_status(job["id"], "processing", f"Loading model for image {i+1} of {num_images}...")

                        # Get the model - this will load it with CPU offloading
                        model = self.model_manager.get_model(job["model_id"])

                        # We've determined that SD3 Pipeline doesn't support callbacks,
                        # so we'll use status updates only before and after generation
                        QueueManager.update_job_status(
                            job["id"],
                            "processing",
                            f"Running generation for image {i+1} of {num_images}..."
                        )

                        # Generate the image - one at a time
                        image = self.model_manager.generate_image(
                            job["model_id"],
                            job["prompt"],
                            negative_prompt=job.get("negative_prompt"),
                            **generation_settings  # Use the settings without num_images
                        )

                        # Prepare metadata for the image
                        metadata = {
                            "model_id": job["model_id"],
                            "prompt": job["prompt"],
                            "negative_prompt": job.get("negative_prompt"),
                            "settings": generation_settings,
                            "generation_time": time.time(),
                            "image_number": i + 1,
                            "total_images": num_images
                        }

                        # Save the image with metadata and job ID
                        image_id = ImageManager.save_image(image, job["id"], metadata)
                        image_ids.append(image_id)
                        print(f"   • Saved image: {image_id}")
                        sys.stdout.flush()

                        # Force memory cleanup between images
                        if i < num_images - 1:
                            # Not the last image, so do cleanup before next generation
                            # --- Comment out cleanup between images ---
                            # self._force_memory_cleanup()
                            # --- --- --- --- --- --- --- --- --- --- ---
                            GenerationPipeline._generation_in_progress = True  # Re-mark as in progress
                            GenerationPipeline._last_model_used = job["model_id"]  # Re-mark model
                finally:
                    # Always mark generation as complete, even if there's an error
                    GenerationPipeline._generation_in_progress = False

                    # Always do cleanup after generation, even if there's an error
                    # --- Comment out cleanup in finally block to allow caching ---
                    # self._force_memory_cleanup()
                    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
                    pass # Keep finally block structure

            # Update job status to completed
            QueueManager.update_job_status(
                job["id"],
                "completed",
                f"Generated {len(image_ids)} image(s)",
                image_ids=image_ids
            )

            print(f"✅ Job {job['id']} completed")
            sys.stdout.flush()

            # Return the first image ID
            return image_ids[0] if image_ids else None

        except Exception as e:
            # Handle generation errors
            error_message = f"Error generating image: {str(e)}"
            print(f"\n❌ {error_message}")
            sys.stdout.flush()

            # Update job status to failed
            QueueManager.update_job_status(job["id"], "failed", error_message)

            # Make sure generation is marked as not in progress
            GenerationPipeline._generation_in_progress = False

            # Force memory cleanup on error
            self._force_memory_cleanup()

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