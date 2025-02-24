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

        return [{
            "id": job["id"],
            "model_id": job["model_id"],
            "prompt": job["prompt"],
            "settings": json.loads(job["settings"]),
            "started_at": job["started_at"]
        } for job in jobs]

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
            "message": job["error_message"],
            "progress": {
                "preparing": job["status"] == "pending",
                "loading_model": job["status"] == "processing" and "Loading model" in (job["error_message"] or ""),
                "generating": job["status"] == "processing" and "Generating" in (job["error_message"] or ""),
                "saving": job["status"] == "processing" and "Saving" in (job["error_message"] or ""),
                "completed": job["status"] == "completed",
                "failed": job["status"] == "failed"
            }
        }

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

        return {
            "id": job["id"],
            "model_id": job["model_id"],
            "prompt": job["prompt"],
            "negative_prompt": job["negative_prompt"],
            "settings": json.loads(job["settings"])
        }

    @staticmethod
    def update_job_status(job_id: str, status: str, message: Optional[str] = None) -> None:
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
        """Retry a failed job by creating a new one with the same parameters"""
        db = get_db()
        try:
            # Get original job details
            job = db.execute(
                """
                SELECT id, model_id, prompt, negative_prompt, settings, status
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
            settings["original_job_id"] = job_id

            # Create a new job with same parameters
            new_job_id = str(uuid.uuid4())
            db.execute(
                """
                INSERT INTO jobs (id, status, model_id, prompt, negative_prompt, settings, error_message)
                VALUES (?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    new_job_id,
                    job["model_id"],
                    job["prompt"],
                    job["negative_prompt"],
                    json.dumps(settings),
                    f"Retry #{retry_count} of job {job_id}"
                )
            )

            # Update original job to reference the retry
            db.execute(
                """
                UPDATE jobs
                SET error_message = ?
                WHERE id = ?
                """,
                (f"Retried as job {new_job_id} (attempt {retry_count}/{max_retries})", job_id)
            )

            db.commit()
            return new_job_id

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to retry job {job_id}: {str(e)}")
            return None