"""
Queue management system for CyberImage
"""
import uuid
import json
from datetime import datetime
from typing import Dict, Optional, List
from flask import current_app
from app.utils.db import get_db
import logging

logger = logging.getLogger(__name__)

class QueueManager:
    """Manages the image generation queue"""

    @staticmethod
    def add_job(model_id: str, prompt: str, settings: Dict) -> str:
        """Add a new job to the queue"""
        db = get_db()
        job_id = str(uuid.uuid4())

        try:
            # Extract negative prompt from settings if present
            negative_prompt = settings.pop("negative_prompt", None)

            db.execute(
                """
                INSERT INTO jobs (id, status, model_id, prompt, negative_prompt, settings)
                VALUES (?, 'pending', ?, ?, ?, ?)
                """,
                (job_id, model_id, prompt, negative_prompt, json.dumps(settings))
            )
            db.commit()
            return job_id
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to add job to queue: {str(e)}")

    @staticmethod
    def get_processing_jobs() -> List[Dict]:
        """Get all jobs currently in processing state"""
        db = get_db()
        jobs = db.execute(
            """
            SELECT id, model_id, prompt, settings, started_at
            FROM jobs
            WHERE status = 'processing'
            ORDER BY started_at ASC
            """
        ).fetchall()

        result = []
        for job in jobs:
            # Create job dict with basic information
            job_dict = {
            "id": job["id"],
            "model_id": job["model_id"],
            "prompt": job["prompt"],
            "settings": json.loads(job["settings"]),
            }

            # Handle started_at with better type information
            if job["started_at"] is not None:
                # Add the original timestamp for debugging
                job_dict["started_at"] = job["started_at"]
                # Log the timestamp type for debugging
                logger.debug(f"Job {job['id']} has timestamp of type {type(job['started_at'])}: {repr(job['started_at'])}")
            else:
                job_dict["started_at"] = None

            result.append(job_dict)

        return result

    @staticmethod
    def get_pending_jobs() -> List[Dict]:
        """Get all pending jobs"""
        db = get_db()
        jobs = db.execute(
            """
            SELECT id, model_id, prompt, settings, created_at
            FROM jobs
            WHERE status = 'pending'
            ORDER BY created_at ASC
            """
        ).fetchall()

        return [{
            "id": job["id"],
            "model_id": job["model_id"],
            "prompt": job["prompt"],
            "settings": json.loads(job["settings"]),
            "created_at": job["created_at"]
        } for job in jobs]

    @staticmethod
    def get_failed_jobs(max_retry_attempts: int = 3) -> List[Dict]:
        """Get all failed jobs that could be retried"""
        db = get_db()
        jobs = db.execute(
            """
            SELECT id, model_id, prompt, negative_prompt, settings, error_message
            FROM jobs
            WHERE status = 'failed'
            ORDER BY created_at DESC
            """
        ).fetchall()

        retryable_jobs = []
        for job in jobs:
            settings = json.loads(job["settings"])
            # Check if the job has exceeded the retry limit
            retry_count = settings.get("retry_count", 0)
            if retry_count < max_retry_attempts:
                retryable_jobs.append({
                    "id": job["id"],
                    "model_id": job["model_id"],
                    "prompt": job["prompt"],
                    "negative_prompt": job["negative_prompt"],
                    "settings": settings,
                    "error_message": job["error_message"],
                    "retry_count": retry_count
                })

        return retryable_jobs

    @staticmethod
    def get_job_status(job_id: str) -> Dict:
        """Get the status of a specific job"""
        db = get_db()
        job = db.execute(
            """
            SELECT id, status, model_id, prompt, negative_prompt, settings,
                   created_at, started_at, completed_at, error_message
            FROM jobs WHERE id = ?
            """,
            (job_id,)
        ).fetchone()

        if job is None:
            return {"error": "Job not found"}

        # Extract error message for parsing
        error_message = job["error_message"] or ""

        # Extract image IDs if they exist in the error_message
        image_ids = []
        if "IMAGE_IDS:" in error_message:
            try:
                # Extract and parse the image IDs
                image_ids_part = error_message.split("IMAGE_IDS:")[1].strip()
                # Remove any text after the JSON
                if " " in image_ids_part:
                    image_ids_part = image_ids_part.split(" ")[0]
                image_ids = json.loads(image_ids_part)

                # Remove the image IDs from the error message
                error_message = error_message.split("IMAGE_IDS:")[0].strip()
            except json.JSONDecodeError:
                # If JSON decoding fails, it means image IDs weren't stored correctly
                # or the message format is unexpected. Silently ignore for status checks.
                pass
            except Exception as e:
                # Log other unexpected errors during parsing
                logger.error(f"Unexpected error parsing image IDs from message for job {job['id']}: {e}", exc_info=True)

        # Determine progress state more accurately
        progress = {
            "preparing": False,
            "loading_model": False,
            "generating": False,
            "saving": False,
            "completed": False,
            "failed": False,
            # Add step information if available
            "step": None,
            "total_steps": None
        }

        # Parse detailed progress from the error_message field
        # (which is used to store status messages, not just errors)
        if "Loading model" in error_message:
            progress["loading_model"] = True
        elif "Generating image" in error_message:
            progress["generating"] = True
            # Try to extract step information if available
            try:
                # Look for patterns like "Step 23/30" or similar in the message
                if "Step " in error_message:
                    step_part = error_message.split("Step ")[1].strip()
                    if "/" in step_part:
                        current_step, total_steps = step_part.split("/", 1)
                        # Remove any text after the number
                        if " " in total_steps:
                            total_steps = total_steps.split(" ")[0]
                        progress["step"] = int(current_step)
                        progress["total_steps"] = int(total_steps)
            except Exception as e:
                logger.warning(f"Error parsing step information: {e}")
        elif "Saving" in error_message:
            progress["saving"] = True
        elif job["status"] == "pending":
            progress["preparing"] = True
        elif job["status"] == "completed":
            progress["completed"] = True
        elif job["status"] == "failed":
            progress["failed"] = True

        # Set default step if we're generating but don't have step info
        if progress["generating"] and progress["step"] is None:
            # Default values based on typical generation settings
            settings = json.loads(job["settings"])
            default_steps = settings.get("num_inference_steps", 30)
            progress["total_steps"] = default_steps
            # Use a sensible default for current step if we can't determine it
            progress["step"] = 1

        result = {
            "id": job["id"],
            "status": job["status"],
            "model_id": job["model_id"],
            "prompt": job["prompt"],
            "negative_prompt": job["negative_prompt"],
            "settings": json.loads(job["settings"]),
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "message": error_message,
            "progress": progress
        }

        # Add image IDs to the result if present
        if image_ids:
            result["image_ids"] = image_ids

        return result

    @staticmethod
    def get_queue_status() -> Dict:
        """Get the current status of the queue"""
        db = get_db()
        stats = db.execute(
            """
            SELECT status, COUNT(*) as count
            FROM jobs
            GROUP BY status
            """
        ).fetchall()

        status_counts = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        }

        for stat in stats:
            status_counts[stat["status"]] = stat["count"]

        return {
            "pending": status_counts["pending"],
            "processing": status_counts["processing"],
            "completed": status_counts["completed"],
            "failed": status_counts["failed"],
            "total": sum(status_counts.values())
        }

    @staticmethod
    def get_next_job() -> Optional[Dict]:
        """Get the next pending job"""
        db = get_db()
        job = db.execute(
            """
            SELECT id, model_id, prompt, negative_prompt, settings
            FROM jobs
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()

        if job is None:
            return None

        settings = json.loads(job["settings"])

        # Store retry info separately in metadata
        metadata = {
            "retry_count": settings.pop("retry_count", 0),
            "original_job_id": settings.pop("original_job_id", None)
        }

        return {
            "id": job["id"],
            "model_id": job["model_id"],
            "prompt": job["prompt"],
            "negative_prompt": job["negative_prompt"],
            "settings": settings,
            "metadata": metadata
        }

    @staticmethod
    def update_job_status(job_id: str, status: str, message: Optional[str] = None, image_ids: List[str] = None) -> None:
        """Update the status of a job"""
        db = get_db()
        now = datetime.utcnow()

        updates = {
            "pending": {"field": None, "status": "pending"},
            "processing": {"field": "started_at", "status": "processing"},
            "completed": {"field": "completed_at", "status": "completed"},
            "failed": {"field": "completed_at", "status": "failed"}
        }

        if status not in updates:
            raise ValueError(f"Invalid status: {status}")

        update = updates[status]
        try:
            # If we have image IDs and the status is completed, save them
            if status == "completed" and image_ids:
                # Store the image IDs in the job record
                # We'll store them in the error_message field as a JSON string
                # prefixed with "IMAGE_IDS:" to distinguish it from actual error messages
                message_with_ids = message or ""
                if message_with_ids:
                    message_with_ids += " "
                message_with_ids += f"IMAGE_IDS:{json.dumps(image_ids)}"

                if update["field"]:
                    db.execute(
                        f"""
                        UPDATE jobs
                        SET status = ?, {update['field']} = ?, error_message = ?
                        WHERE id = ?
                        """,
                        (update["status"], now, message_with_ids, job_id)
                    )
                else:
                    db.execute(
                        """
                        UPDATE jobs
                        SET status = ?, error_message = ?
                        WHERE id = ?
                        """,
                        (update["status"], message_with_ids, job_id)
                    )
            else:
                # Standard update without image IDs
                if update["field"]:
                    db.execute(
                        f"""
                        UPDATE jobs
                        SET status = ?, {update['field']} = ?, error_message = ?
                        WHERE id = ?
                        """,
                        (update["status"], now, message, job_id)
                    )
                else:
                    db.execute(
                        """
                        UPDATE jobs
                        SET status = ?, error_message = ?
                        WHERE id = ?
                        """,
                        (update["status"], message, job_id)
                    )
            db.commit()
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to update job status: {str(e)}")

    @staticmethod
    def cleanup_stalled_jobs(timeout_minutes: int = 30) -> None:
        """Reset stalled processing jobs to pending"""
        db = get_db()
        try:
            db.execute(
                """
                UPDATE jobs
                SET status = 'pending', started_at = NULL, error_message = 'Job reset due to timeout'
                WHERE status = 'processing'
                AND started_at < datetime('now', ?)
                """,
                (f'-{timeout_minutes} minutes',)
            )
            db.commit()
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to cleanup stalled jobs: {str(e)}")

    @staticmethod
    def retry_failed_job(job_id: str, max_retries: int = 3) -> Optional[str]:
        """Retry a failed job by updating its status back to pending"""
        db = get_db()
        try:
            # Get original job details
            job = db.execute(
                """
                SELECT id, model_id, prompt, negative_prompt, settings, status, error_message
                FROM jobs
                WHERE id = ?
                """,
                (job_id,)
            ).fetchone()

            if not job:
                return None

            # Only retry failed jobs
            if job["status"] != "failed":
                return None

            # Check if error is due to incompatible parameters - don't retry these
            error_message = job["error_message"] or ""
            if ("unexpected keyword argument" in error_message or
                "got an unexpected keyword" in error_message or
                "not supported" in error_message or
                "StableDiffusion3Pipeline" in error_message or
                "incompatible" in error_message):
                logger.warning(f"Not retrying job {job_id} due to likely model incompatibility: {error_message}")
                # Update the job with a note that it won't be retried
                db.execute(
                    """
                    UPDATE jobs
                    SET error_message = ?
                    WHERE id = ?
                    """,
                    (f"{error_message} - Automatic retry disabled: model incompatibility detected.", job_id)
                )
                db.commit()
                return None

            # Extract settings and check retry count
            settings = json.loads(job["settings"])

            # Increment retry count or initialize it
            retry_count = settings.get("retry_count", 0) + 1
            if retry_count > max_retries:
                # Update the original job with a note that max retries exceeded
                db.execute(
                    """
                    UPDATE jobs
                    SET error_message = ?
                    WHERE id = ?
                    """,
                    (f"Failed after {max_retries} retry attempts", job_id)
                )
                db.commit()
                return None

            # Update settings with retry information
            settings["retry_count"] = retry_count

            # Instead of creating a new job, update the existing job to pending
            db.execute(
                """
                UPDATE jobs
                SET status = 'pending',
                    started_at = NULL,
                    completed_at = NULL,
                    settings = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (
                    json.dumps(settings),
                    f"Job retried (attempt {retry_count}/{max_retries})",
                    job_id
                )
            )

            db.commit()
            logger.info(f"Job {job_id} updated back to pending status (retry attempt {retry_count}/{max_retries})")
            return job_id

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to retry job {job_id}: {str(e)}")
            return None

    @staticmethod
    def get_all_jobs() -> List[Dict]:
        """Get all jobs in the queue with detailed information"""
        db = get_db()
        jobs = db.execute(
            """
            SELECT id, status, model_id, prompt, negative_prompt, settings,
                   created_at, started_at, completed_at, error_message
            FROM jobs
            ORDER BY created_at DESC
            """
        ).fetchall()

        return [{
            "id": job["id"],
            "status": job["status"],
            "model_id": job["model_id"],
            "prompt": job["prompt"],
            "negative_prompt": job["negative_prompt"],
            "settings": json.loads(job["settings"]),
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "error_message": job["error_message"]
        } for job in jobs]

    @staticmethod
    def get_job(job_id: str) -> Optional[Dict]:
        """Get detailed information for a specific job"""
        db = get_db()
        job = db.execute(
            """
            SELECT id, status, model_id, prompt, negative_prompt, settings,
                   created_at, started_at, completed_at, error_message
            FROM jobs WHERE id = ?
            """,
            (job_id,)
        ).fetchone()

        if job is None:
            return None

        return {
            "id": job["id"],
            "status": job["status"],
            "model_id": job["model_id"],
            "prompt": job["prompt"],
            "negative_prompt": job["negative_prompt"],
            "settings": json.loads(job["settings"]),
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "error_message": job["error_message"]
        }

    @staticmethod
    def delete_job(job_id: str) -> bool:
        """Delete a job from the queue"""
        db = get_db()
        try:
            # Check if job exists
            job = db.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                return False

            # Delete the job
            db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to delete job {job_id}: {str(e)}")
            return False

    @staticmethod
    def clear_queue_by_status(status: str) -> int:
        """Clear all jobs with a specific status from the queue"""
        db = get_db()
        try:
            # Get count of affected rows
            count = db.execute("SELECT COUNT(*) as count FROM jobs WHERE status = ?", (status,)).fetchone()["count"]

            # Delete the jobs
            db.execute("DELETE FROM jobs WHERE status = ?", (status,))
            db.commit()
            return count
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to clear queue with status {status}: {str(e)}")
            return 0

    @staticmethod
    def clear_all_jobs() -> int:
        """Clear all jobs from the queue"""
        db = get_db()
        try:
            # Get count of affected rows
            count = db.execute("SELECT COUNT(*) as count FROM jobs").fetchone()["count"]

            # Delete all jobs
            db.execute("DELETE FROM jobs")
            db.commit()
            return count
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to clear all jobs: {str(e)}")
            return 0

    @staticmethod
    def reset_stalled_jobs() -> int:
        """Reset any stalled processing jobs back to pending state

        Returns the number of jobs that were reset
        """
        db = get_db()

        # First get all processing jobs
        processing_jobs = db.execute(
            """
            SELECT id
            FROM jobs
            WHERE status = 'processing'
            """
        ).fetchall()

        if not processing_jobs:
            return 0

        reset_count = 0
        for job in processing_jobs:
            job_id = job["id"]
            try:
                # Reset the job to pending
                db.execute(
                    """
                    UPDATE jobs
                    SET status = 'pending', started_at = NULL, error_message = 'Job reset after being stalled'
                    WHERE id = ?
                    """,
                    (job_id,)
                )
                db.commit()
                reset_count += 1
                logger.info(f"Reset stalled job {job_id} to pending status")
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to reset stalled job {job_id}: {str(e)}")

        logger.info(f"Reset {reset_count} stalled jobs to pending status")
        return reset_count