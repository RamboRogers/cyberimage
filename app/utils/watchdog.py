"""
GPU watchdog for monitoring job execution and recovery
"""
import threading
import time
import torch
import gc
import sys
import logging
from app.utils.queue import QueueManager

# Configure logging
logger = logging.getLogger(__name__)

class GPUWatchdog:
    """
    Monitors job execution and provides recovery mechanisms for stalled jobs.
    Note: High GPU memory usage (up to 100%) is normal and expected for Flux models.
    """

    def __init__(self, model_manager, max_stalled_time=400, check_interval=30,
                 recovery_cooldown=300):
        """
        Initialize the GPU watchdog

        Args:
            model_manager: The ModelManager instance to monitor
            max_stalled_time: Maximum time (in seconds) a job can be "processing" before recovery
            check_interval: Time (in seconds) between health checks
            recovery_cooldown: Minimum time (in seconds) between emergency recoveries
        """
        self.model_manager = model_manager
        self.max_stalled_time = max_stalled_time
        self.check_interval = check_interval
        self.recovery_cooldown = recovery_cooldown

        self.running = False
        self.thread = None
        self.last_recovery_time = 0
        self.recovery_in_progress = False

        print("\n🔍 GPU Watchdog initialized")
        print(f"   • Max stalled job time: {max_stalled_time} seconds")
        print(f"   • Check interval: {check_interval} seconds")
        print(f"   • Recovery cooldown: {recovery_cooldown} seconds")
        sys.stdout.flush()

    def start(self):
        """Start the watchdog monitoring thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print("\n🚀 GPU Watchdog started")
        sys.stdout.flush()

    def stop(self):
        """Stop the watchdog monitoring thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        print("\n🛑 GPU Watchdog stopped")
        sys.stdout.flush()

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Log GPU memory status (for information only)
                self._log_gpu_memory()

                # Check for stalled jobs - this is our primary focus
                self._check_stalled_jobs()

                # Sleep until next check
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"\n❌ Error in watchdog monitoring: {str(e)}")
                sys.stdout.flush()
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

            # Log memory state at regular intervals
            print(f"\n🔍 Watchdog GPU Memory Check:")
            print(f"   • Total Memory: {total_memory:.2f}GB")
            print(f"   • Allocated Memory: {memory_allocated:.2f}GB")
            print(f"   • Reserved Memory: {memory_reserved:.2f}GB")
            print(f"   • Usage: {usage_percentage:.2%}")
            sys.stdout.flush()

            # Note: We don't trigger recovery based on memory usage alone
            # since 100% usage is normal for Flux models
        except Exception as e:
            print(f"\n❌ Error checking GPU memory: {str(e)}")
            sys.stdout.flush()
            logger.error(f"Error checking GPU memory: {str(e)}")

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
                # If this is already a timestamp, use it directly
                start_time = job.get("started_at")
                if isinstance(start_time, str):
                    # If it's an ISO format timestamp, convert to time
                    try:
                        from datetime import datetime
                        # This assumes the timestamp is in iso format
                        dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
                        start_time = dt.timestamp()
                    except Exception as e:
                        print(f"\n❌ Error parsing job timestamp: {str(e)}")
                        sys.stdout.flush()
                        continue

                # Calculate how long the job has been processing
                processing_time = current_time - start_time

                # If it's been processing too long, attempt recovery
                if processing_time > self.max_stalled_time:
                    print(f"\n⚠️ Stalled job detected: {job['id']}")
                    print(f"   • Processing time: {processing_time:.1f} seconds")
                    print(f"   • Exceeds maximum allowed time: {self.max_stalled_time} seconds")
                    sys.stdout.flush()
                    self._recover_stalled_job(job)
        except Exception as e:
            print(f"\n❌ Error checking stalled jobs: {str(e)}")
            sys.stdout.flush()
            logger.error(f"Error checking stalled jobs: {str(e)}")

    def _recover_stalled_job(self, job):
        """Mark a stalled job as failed and resubmit it for retry"""
        try:
            job_id = job["id"]
            print(f"\n🔄 Recovering stalled job: {job_id}")
            sys.stdout.flush()

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
            new_job_id = QueueManager.retry_failed_job(job_id)
            if new_job_id:
                print(f"✅ Job {job_id} automatically retried as {new_job_id}")
            else:
                print(f"❌ Could not retry job {job_id} - may have exceeded retry limit")

            print(f"✅ Recovery complete for job: {job_id}")
            sys.stdout.flush()
            logger.info(f"Recovered stalled job: {job_id}")
        except Exception as e:
            print(f"\n❌ Error recovering stalled job: {str(e)}")
            sys.stdout.flush()
            logger.error(f"Error recovering stalled job: {str(e)}")

    def _perform_emergency_recovery(self):
        """Handle critical situations by cleaning up resources"""
        if self.recovery_in_progress:
            print("\n⚠️ Emergency recovery already in progress, skipping")
            sys.stdout.flush()
            return

        current_time = time.time()
        if current_time - self.last_recovery_time < self.recovery_cooldown:
            cooldown_remaining = self.recovery_cooldown - (current_time - self.last_recovery_time)
            print(f"\n⚠️ Recovery cooldown period active. {cooldown_remaining:.0f}s remaining.")
            sys.stdout.flush()
            return

        self.recovery_in_progress = True
        self.last_recovery_time = current_time

        try:
            print("\n🚨 EMERGENCY RECOVERY INITIATED")
            print("   • Forcing complete cleanup of resources")
            sys.stdout.flush()

            # First, mark all currently processing jobs as failed
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

                print(f"\n✅ Emergency recovery completed")
                print(f"   • GPU memory after recovery: {memory_allocated:.2f}GB ({usage_percentage:.2%})")
                sys.stdout.flush()
        except Exception as e:
            print(f"\n❌ Error during emergency recovery: {str(e)}")
            sys.stdout.flush()
            logger.error(f"Error during emergency recovery: {str(e)}")
        finally:
            self.recovery_in_progress = False

    def _recover_all_processing_jobs(self):
        """Mark all currently processing jobs as failed during emergency recovery and retry them"""
        try:
            processing_jobs = QueueManager.get_processing_jobs()
            if not processing_jobs:
                print("   • No processing jobs to recover")
                return

            print(f"   • Recovering {len(processing_jobs)} processing jobs")
            for job in processing_jobs:
                job_id = job["id"]
                QueueManager.update_job_status(
                    job_id,
                    "failed",
                    "Job interrupted by emergency recovery"
                )
                print(f"   • Marked job {job_id} as failed")

                # Try to retry the job
                new_job_id = QueueManager.retry_failed_job(job_id)
                if new_job_id:
                    print(f"   • Job {job_id} automatically retried as {new_job_id}")
                else:
                    print(f"   • Could not retry job {job_id} - may have exceeded retry limit")

            print(f"   • All processing jobs marked as failed for recovery")
            sys.stdout.flush()
        except Exception as e:
            print(f"\n❌ Error recovering processing jobs: {str(e)}")
            sys.stdout.flush()
            logger.error(f"Error recovering processing jobs: {str(e)}")