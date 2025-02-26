from flask import render_template, jsonify, request, redirect, url_for, current_app
from app.admin import bp
from app.utils.queue import QueueManager
from app.utils.image import ImageManager
from app.models import AVAILABLE_MODELS
import json
from datetime import datetime
from app.utils.db import get_db

@bp.route('/admin')
def admin_panel():
    """Render the admin panel page"""
    # Get all jobs from the queue
    jobs = QueueManager.get_all_jobs()
    queue_status = QueueManager.get_queue_status()

    # Get generator status
    try:
        from app.models.generator import GenerationPipeline
        generator = GenerationPipeline()
        generator_status = {
            "is_running": generator.is_running,
            "is_main_process": generator.is_main_process
        }
    except Exception:
        generator_status = {
            "is_running": False,
            "is_main_process": False
        }

    return render_template('admin.html',
                         jobs=jobs,
                         queue_status=queue_status,
                         generator_status=generator_status,
                         available_models=AVAILABLE_MODELS)

@bp.route('/admin/queue/clear', methods=['POST'])
def clear_queue():
    """Clear jobs from the queue with the specified status"""
    status = request.form.get('status', None)
    job_id = request.form.get('job_id', None)

    if job_id:
        # Delete a specific job
        QueueManager.delete_job(job_id)
        return jsonify({"success": True, "message": f"Job {job_id} deleted successfully"})
    elif status:
        # Delete all jobs with the specified status
        count = QueueManager.clear_queue_by_status(status)
        return jsonify({"success": True, "message": f"Cleared {count} jobs with status '{status}'"})
    else:
        # Delete all jobs (careful with this!)
        count = QueueManager.clear_all_jobs()
        return jsonify({"success": True, "message": f"Cleared {count} jobs from the queue"})

@bp.route('/admin/job/<job_id>')
def job_details(job_id):
    """Get details for a specific job"""
    job = QueueManager.get_job(job_id)
    images = ImageManager.get_images_for_job(job_id) if job else []

    # For AJAX requests return JSON
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({
            "job": job,
            "images": images
        })

    # For regular requests, render the template
    return render_template('admin.html',
                         job=job,
                         images=images,
                         available_models=AVAILABLE_MODELS)

@bp.route('/admin/queue/retry', methods=['POST'])
def retry_job():
    """Retry a failed job (admin-initiated retries always force retry)"""
    job_id = request.form.get('job_id')

    if not job_id:
        return jsonify({
            "success": False,
            "message": "No job ID provided"
        })

    # Get job details before retry for logging
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
            return jsonify({
                "success": False,
                "message": f"Job {job_id} not found"
            })

        # Only retry failed jobs
        if job["status"] != "failed":
            return jsonify({
                "success": False,
                "message": f"Job {job_id} is not in failed state (current state: {job['status']})"
            })

        # Extract settings
        settings = json.loads(job["settings"])

        # Reset retry count to 0 - all admin retries are force retries
        settings["retry_count"] = 0

        # Log this admin retry
        settings["admin_retried"] = True
        settings["admin_retry_timestamp"] = datetime.now().isoformat()

        # Update the job to pending
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
                "Job retried by admin",
                job_id
            )
        )

        db.commit()
        current_app.logger.info(f"Admin panel: Retried job {job_id} (retry count reset)")

        return jsonify({
            "success": True,
            "message": f"Job {job_id} has been retried with retry count reset"
        })

    except Exception as e:
        db.rollback()
        current_app.logger.error(f"Error retrying job {job_id}: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error retrying job: {str(e)}"
        })

@bp.route('/admin/queue/retry-all', methods=['POST'])
def retry_all_failed_jobs():
    """Retry all failed jobs (admin-initiated retries always force retry)"""
    db = get_db()

    try:
        # Get all failed jobs
        jobs = db.execute(
            """
            SELECT id, model_id, prompt, negative_prompt, settings, error_message
            FROM jobs
            WHERE status = 'failed'
            ORDER BY created_at DESC
            """
        ).fetchall()

        if not jobs:
            return jsonify({
                "success": True,
                "message": "No failed jobs to retry"
            })

        # Log the attempt
        current_app.logger.info(f"Admin panel: Retrying all {len(jobs)} failed jobs")

        # Try to retry each job
        retry_count = 0
        retry_results = []

        for job in jobs:
            job_id = job["id"]
            retry_result = {
                "job_id": job_id,
                "prompt": job["prompt"][:50] + "..." if len(job["prompt"]) > 50 else job["prompt"],
                "model": job["model_id"],
                "success": False
            }

            try:
                # Extract settings
                settings = json.loads(job["settings"])

                # Reset retry count to 0 - all admin retries are force retries
                settings["retry_count"] = 0

                # Log this admin retry
                settings["admin_retried"] = True
                settings["admin_retry_timestamp"] = datetime.now().isoformat()

                # Update the job to pending
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
                        "Job retried by admin",
                        job_id
                    )
                )

                retry_count += 1
                retry_result["success"] = True
                current_app.logger.info(f"Successfully retried job {job_id}")

            except Exception as e:
                retry_result["error"] = str(e)
                current_app.logger.error(f"Error retrying job {job_id}: {str(e)}")

            retry_results.append(retry_result)

        # Commit all changes
        db.commit()

        # Log summary
        current_app.logger.info(f"Bulk retry complete: {retry_count}/{len(jobs)} jobs successfully queued")

        return jsonify({
            "success": True,
            "message": f"Retried {retry_count} failed jobs",
            "details": {
                "total_failed": len(jobs),
                "total_retried": retry_count,
                "results": retry_results
            }
        })

    except Exception as e:
        db.rollback()
        current_app.logger.error(f"Error during bulk retry: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error retrying all jobs: {str(e)}"
        })

@bp.route('/admin/queue/force-retry', methods=['POST'])
def force_retry_job():
    """Force retry a failed job regardless of retry count"""
    job_id = request.form.get('job_id')

    if not job_id:
        return jsonify({
            "success": False,
            "message": "No job ID provided"
        })

    # Get job details before retry for logging
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
            return jsonify({
                "success": False,
                "message": f"Job {job_id} not found"
            })

        # Only retry failed jobs
        if job["status"] != "failed":
            return jsonify({
                "success": False,
                "message": f"Job {job_id} is not in failed state (current state: {job['status']})"
            })

        # Extract settings
        settings = json.loads(job["settings"])

        # Reset retry count to 0
        settings["retry_count"] = 0

        # Log this force retry
        settings["force_retried"] = True
        settings["force_retry_timestamp"] = datetime.now().isoformat()

        # Update the job to pending
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
                "Job force-retried by admin",
                job_id
            )
        )

        db.commit()
        current_app.logger.info(f"Admin panel: Force-retried job {job_id} (retry count reset)")

        return jsonify({
            "success": True,
            "message": f"Job {job_id} has been force-retried with retry count reset"
        })

    except Exception as e:
        db.rollback()
        current_app.logger.error(f"Error force-retrying job {job_id}: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error force-retrying job: {str(e)}"
        })

@bp.route('/admin/queue/force-retry-all', methods=['POST'])
def force_retry_all_failed_jobs():
    """Force retry all failed jobs regardless of retry count"""
    db = get_db()

    try:
        # Get all failed jobs
        jobs = db.execute(
            """
            SELECT id, model_id, prompt, negative_prompt, settings, error_message
            FROM jobs
            WHERE status = 'failed'
            ORDER BY created_at DESC
            """
        ).fetchall()

        if not jobs:
            return jsonify({
                "success": True,
                "message": "No failed jobs to retry"
            })

        # Log the attempt
        current_app.logger.info(f"Admin panel: Force-retrying all {len(jobs)} failed jobs")

        # Try to retry each job
        retry_count = 0
        retry_results = []

        for job in jobs:
            job_id = job["id"]
            retry_result = {
                "job_id": job_id,
                "prompt": job["prompt"][:50] + "..." if len(job["prompt"]) > 50 else job["prompt"],
                "model": job["model_id"],
                "success": False
            }

            try:
                # Extract settings
                settings = json.loads(job["settings"])

                # Reset retry count to 0
                settings["retry_count"] = 0

                # Log this force retry
                settings["force_retried"] = True
                settings["force_retry_timestamp"] = datetime.now().isoformat()

                # Update the job to pending
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
                        "Job force-retried by admin",
                        job_id
                    )
                )

                retry_count += 1
                retry_result["success"] = True
                current_app.logger.info(f"Successfully force-retried job {job_id}")

            except Exception as e:
                retry_result["error"] = str(e)
                current_app.logger.error(f"Error force-retrying job {job_id}: {str(e)}")

            retry_results.append(retry_result)

        # Commit all changes
        db.commit()

        # Log summary
        current_app.logger.info(f"Bulk force-retry complete: {retry_count}/{len(jobs)} jobs successfully queued")

        return jsonify({
            "success": True,
            "message": f"Force-retried {retry_count} failed jobs",
            "details": {
                "total_failed": len(jobs),
                "total_retried": retry_count,
                "results": retry_results
            }
        })

    except Exception as e:
        db.rollback()
        current_app.logger.error(f"Error during bulk force-retry: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error force-retrying all jobs: {str(e)}"
        })

@bp.route('/admin/queue/restart-generator', methods=['POST'])
def restart_generator():
    """Force restart the generation pipeline to process pending jobs"""
    try:
        # Get the generator instance
        from app.models.generator import GenerationPipeline
        generator = GenerationPipeline()

        # Force re-initialization
        generator.is_running = False
        generator._initialize()

        # Trigger queue processing
        generator._queue_event.set()

        # Get current queue status
        queue_status = QueueManager.get_queue_status()
        pending_count = queue_status.get("pending", 0)

        current_app.logger.info(f"Admin panel: Generator pipeline restarted. {pending_count} pending jobs in queue.")

        return jsonify({
            "success": True,
            "message": f"Generator pipeline restarted. {pending_count} pending jobs in queue.",
            "is_running": generator.is_running,
            "is_main_process": generator.is_main_process
        })
    except Exception as e:
        current_app.logger.error(f"Error restarting generator: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"Error restarting generator: {str(e)}"
        })