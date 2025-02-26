"""
GPU watchdog for monitoring job execution and recovery
"""
import threading
import time
import torch
import gc
import sys
import logging
from flask import current_app
from app.utils.queue import QueueManager
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class GPUWatchdog:
    """
    Monitors job execution and provides recovery mechanisms for stalled jobs.
    Note: High GPU memory usage (up to 100%) is normal and expected for Flux models.
    """

    def __init__(self, model_manager, app=None, max_stalled_time=400, check_interval=30,
                 recovery_cooldown=300):
        """
        Initialize the GPU watchdog

        Args:
            model_manager: The ModelManager instance to monitor
            app: Flask application instance for context
            max_stalled_time: Maximum time (in seconds) a job can be "processing" before recovery
            check_interval: Time (in seconds) between health checks
            recovery_cooldown: Minimum time (in seconds) between emergency recoveries
        """
        self.model_manager = model_manager
        self.app = app
        self.max_stalled_time = max_stalled_time
        self.check_interval = check_interval
        self.recovery_cooldown = recovery_cooldown

        self.running = False
        self.thread = None
        self.last_recovery_time = 0
        self.recovery_in_progress = False

        logger.info(f"GPU Watchdog initialized (max_stalled_time={max_stalled_time}s, check_interval={check_interval}s)")

    def _get_app(self):
        """Get the Flask application instance or the stored one"""
        return self.app or current_app._get_current_object()

    def start(self):
        """Start the watchdog monitoring thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("GPU Watchdog started")

    def stop(self):
        """Stop the watchdog monitoring thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("GPU Watchdog stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Log GPU memory status (for information only)
                self._log_gpu_memory()

                # Check for stalled jobs - this is our primary focus
                with self._get_app().app_context():
                    self._check_stalled_jobs()

                # Sleep until next check
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Watchdog error: {str(e)}")
                # Sleep a bit longer on error
                time.sleep(self.check_interval * 2)

    def _log_gpu_memory(self):
        """Log GPU memory usage for monitoring purposes (no recovery action)"""
        if not torch.cuda.is_available():
            return  # Skip if not using CUDA

        try:
            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Calculate usage percentage
            usage_percentage = memory_allocated / total_memory

            # Only log if usage is above 90% or below 10% to reduce chattiness
            if usage_percentage > 0.9 or usage_percentage < 0.1:
                logger.info(f"GPU Memory: {memory_allocated:.2f}GB/{total_memory:.2f}GB ({usage_percentage:.2%})")

            # Don't print to stdout for regular checks
        except Exception as e:
            logger.error(f"Error checking GPU memory: {str(e)}")
            # Don't print to stdout

    def _check_stalled_jobs(self):
        """Check for jobs that have been processing for too long"""
        try:
            processing_jobs = QueueManager.get_processing_jobs()
            current_time = time.time()

            for job in processing_jobs:
                # Skip if no started_at time
                if not job.get("started_at"):
                    continue

                # Parse the timestamp from the database
                start_time = job.get("started_at")

                # Handle different types of start_time values with robust parsing
                try:
                    # Case 1: Already a timestamp (float or int)
                    if isinstance(start_time, (int, float)):
                        pass  # Already in the correct format

                    # Case 2: Already a datetime object
                    elif isinstance(start_time, datetime):
                        start_time = start_time.timestamp()

                    # Case 3: String format - try multiple parsing approaches
                    elif isinstance(start_time, str):
                        # Try parsing as ISO format first
                        try:
                            # Handle various ISO formats with/without timezone
                            if 'T' in start_time:
                                # Standard ISO format
                                dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            else:
                                # SQLite date format
                                dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                            start_time = dt.timestamp()
                        except ValueError:
                            # Try other common formats
                            for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%fZ']:
                                try:
                                    dt = datetime.strptime(start_time, fmt)
                                    start_time = dt.timestamp()
                                    break
                                except ValueError:
                                    continue
                            else:
                                # If we get here, none of the formats worked
                                raise ValueError(f"Could not parse timestamp: {start_time}")

                    # Case 4: Unhandled type
                    else:
                        logger.warning(f"Unhandled timestamp type: {type(start_time)}")
                        continue

                    # Calculate how long the job has been processing
                    processing_time = current_time - start_time

                    # If it's been processing too long, attempt recovery
                    if processing_time > self.max_stalled_time:
                        logger.warning(f"Stalled job detected: {job['id']} (processing for {processing_time:.1f}s)")
                        self._recover_stalled_job(job)

                except Exception as e:
                    logger.warning(f"Error processing job timestamp for job {job['id']}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error checking stalled jobs: {str(e)}")

    def _recover_stalled_job(self, job):
        """Mark a stalled job as failed and resubmit it for retry"""
        try:
            job_id = job["id"]
            logger.info(f"Recovering stalled job: {job_id}")

            # Mark job as failed
            QueueManager.update_job_status(
                job_id,
                "failed",
                "Job failed due to timeout - will be retried automatically"
            )

            # Force model cleanup
            if self.model_manager:
                self._perform_emergency_recovery()

            # Now retry the job
            retried_job_id = QueueManager.retry_failed_job(job_id)
            if retried_job_id:
                logger.info(f"Job {job_id} reset to pending state and will be retried")
            else:
                logger.warning(f"Could not retry job {job_id} - may have exceeded retry limit")

        except Exception as e:
            logger.error(f"Error recovering stalled job: {str(e)}")

    def _perform_emergency_recovery(self):
        """Handle critical situations by cleaning up resources"""
        if self.recovery_in_progress:
            logger.info("Emergency recovery already in progress, skipping")
            return

        # Use time.time() directly to avoid any datetime conversion issues
        current_time = time.time()

        # Ensure last_recovery_time is a valid timestamp
        if not isinstance(self.last_recovery_time, (int, float)):
            logger.warning(f"Invalid last_recovery_time ({type(self.last_recovery_time)}), resetting to 0")
            self.last_recovery_time = 0

        if current_time - self.last_recovery_time < self.recovery_cooldown:
            # Don't log cooldown periods to reduce noise
            return

        self.recovery_in_progress = True
        self.last_recovery_time = current_time

        try:
            logger.warning("EMERGENCY RECOVERY INITIATED")

            # First, mark all currently processing jobs as failed
            with self._get_app().app_context():
                self._recover_all_processing_jobs()

            # Force model cleanup
            if self.model_manager:
                self.model_manager._force_memory_cleanup()

            # Additional cleanup steps
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()

                # Check memory after cleanup
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                usage_percentage = memory_allocated / total_memory

                logger.info(f"Emergency recovery completed. GPU memory: {memory_allocated:.2f}GB ({usage_percentage:.2%})")
        except Exception as e:
            logger.error(f"Error during emergency recovery: {str(e)}")
        finally:
            self.recovery_in_progress = False

    def _recover_all_processing_jobs(self):
        """Mark all currently processing jobs as failed during emergency recovery and retry them"""
        try:
            processing_jobs = QueueManager.get_processing_jobs()
            if not processing_jobs:
                logger.info("No processing jobs to recover")
                return

            logger.info(f"Recovering {len(processing_jobs)} processing jobs")
            for job in processing_jobs:
                job_id = job["id"]
                QueueManager.update_job_status(
                    job_id,
                    "failed",
                    "Job interrupted by emergency recovery"
                )

                # Try to retry the job
                retried_job_id = QueueManager.retry_failed_job(job_id)
                if not retried_job_id:
                    logger.warning(f"Could not retry job {job_id} - may have exceeded retry limit")

        except Exception as e:
            logger.error(f"Error recovering processing jobs: {str(e)}")