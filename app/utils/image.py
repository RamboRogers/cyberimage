"""
Image utilities for CyberImage
"""
import os
import uuid
import json
from datetime import datetime
import pytz
from typing import Dict, Optional, List, Union
from PIL import Image
from flask import current_app
from app.utils.db import get_db

class ImageManager:
    """Manages image storage and retrieval"""

    @staticmethod
    def _convert_to_local(utc_dt) -> datetime:
        """Convert UTC datetime to local time"""
        if not utc_dt:
            return None

        # Ensure UTC timezone is set
        if utc_dt.tzinfo is None:
            utc_dt = pytz.UTC.localize(utc_dt)

        # Convert to local timezone
        local_tz = datetime.now().astimezone().tzinfo
        return utc_dt.astimezone(local_tz)

    @staticmethod
    def save_image(image: Optional[Image.Image], job_id: str, metadata: Dict, image_id: Optional[str] = None, file_path: Optional[str] = None) -> str:
        """Save an image/video record in the database. If image is provided, save it.

        Args:
            image: PIL Image object (optional, provide None if file already saved, e.g., video)
            job_id: The ID of the job that generated the media.
            metadata: Dictionary containing metadata about the media.
            image_id: Optional pre-generated ID for the media record.
            file_path: Optional pre-generated relative file path for the media record.

        Returns:
            The ID of the saved media record.
        """
        db = get_db()
        # Use provided ID or generate a new one
        media_id = image_id if image_id else str(uuid.uuid4())

        # Get job information for metadata (optional, enhance metadata if job exists)
        try:
            job = db.execute(
                """
                    SELECT model_id, prompt, settings
                FROM jobs WHERE id = ?
                """,
                (job_id,)
            ).fetchone()

            if job:
                job_settings = json.loads(job["settings"])
                # Update metadata with job info, ensure settings are merged/overwritten correctly
                metadata.update({
                    "model_id": metadata.get("model_id", job["model_id"]), # Prioritize metadata's model_id
                    "prompt": metadata.get("prompt", job["prompt"]),       # Prioritize metadata's prompt
                    # Merge settings, prioritizing specific metadata settings over job settings
                    "settings": {**job_settings, **metadata.get("settings", {})}
                })
            else:
                current_app.logger.warning(f"Job {job_id} not found when saving media {media_id}")
        except Exception as e:
            current_app.logger.error(f"Error fetching job info {job_id} for media {media_id}: {e}")

        relative_path = None
        full_save_path = None

        if file_path:
            # Use pre-generated relative path
            relative_path = file_path
            full_save_path = os.path.join(current_app.config["IMAGES_PATH"], relative_path)
            current_app.logger.debug(f"Using pre-defined path for media {media_id}: {relative_path}")
        elif image:
            # Generate path and save the image file
            today = datetime.utcnow().strftime("%Y/%m/%d")
            image_dir = os.path.join(current_app.config["IMAGES_PATH"], today)
            os.makedirs(image_dir, exist_ok=True)

            file_name = f"{media_id}.png"
            relative_path = os.path.join(today, file_name)
            full_save_path = os.path.join(image_dir, file_name)
            current_app.logger.debug(f"Saving new image for media {media_id} to: {relative_path}")
            try:
                image.save(full_save_path, "PNG")
            except Exception as save_err:
                raise Exception(f"Failed to save image file to {full_save_path}: {save_err}")
        else:
            # Error: No image provided and no file_path provided
            raise ValueError("ImageManager.save_image requires either an image object or a file_path.")

        # Record in database
        try:
            db.execute(
                """
                INSERT INTO images (id, job_id, file_path, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (media_id, job_id, relative_path, json.dumps(metadata))
            )
            db.commit()
            current_app.logger.info(f"Saved media record {media_id} to database.")
            return media_id
        except Exception as e:
            db.rollback()
            # Clean up the image file only if we created it in this function call
            if full_save_path and not file_path and os.path.exists(full_save_path):
                current_app.logger.warning(f"Rolling back DB commit, deleting created file: {full_save_path}")
                os.remove(full_save_path)
            raise Exception(f"Failed to save media record {media_id} to database: {str(e)}")

    @staticmethod
    def get_image_path(image_id: str) -> Optional[str]:
        """Get the full path to an image"""
        db = get_db()
        result = db.execute(
            "SELECT file_path FROM images WHERE id = ?",
            (image_id,)
        ).fetchone()

        if result is None:
            return None

        return os.path.join(current_app.config["IMAGES_PATH"], result["file_path"])

    @staticmethod
    def get_image_metadata(image_id: str) -> Optional[Dict]:
        """Get metadata for an image"""
        db = get_db()
        result = db.execute(
            "SELECT metadata FROM images WHERE id = ?",
            (image_id,)
        ).fetchone()

        if result is None:
            return None

        return json.loads(result["metadata"])

    @staticmethod
    def get_image_info(image_id: str) -> Optional[Dict]:
        """Get complete information for a single image by ID"""
        db = get_db()
        img = db.execute(
            """
            SELECT id, file_path, job_id, created_at, metadata
            FROM images
            WHERE id = ?
            """,
            (image_id,)
        ).fetchone()

        if img is None:
            return None

        return {
            "id": img["id"],
            "file_path": img["file_path"],
            "job_id": img["job_id"],
            "created_at": ImageManager._convert_to_local(img["created_at"]),
            "metadata": json.loads(img["metadata"])
        }

    @staticmethod
    def get_job_images(job_id: str) -> list:
        """Get all images associated with a job"""
        db = get_db()
        images = db.execute(
            """
            SELECT id, file_path, created_at, metadata
            FROM images
            WHERE job_id = ?
            ORDER BY created_at DESC
            """,
            (job_id,)
        ).fetchall()

        return [{
            "id": img["id"],
            "file_path": img["file_path"],
            "created_at": ImageManager._convert_to_local(img["created_at"]),
            "metadata": json.loads(img["metadata"])
        } for img in images]

    @staticmethod
    def get_recent_images(limit: int = 12) -> List[Dict]:
        """Get recent images with metadata"""
        db = get_db()
        images = db.execute(
            """
            SELECT id, file_path, created_at, metadata
            FROM images
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

        return [{
            "id": img["id"],
            "created_at": img["created_at"],  # SQLite returns datetime object directly
            "metadata": json.loads(img["metadata"]),
            "model_id": json.loads(img["metadata"])["model_id"],
            "prompt": json.loads(img["metadata"])["prompt"]
        } for img in images]

    @staticmethod
    def get_all_images(page: int = 1, per_page: int = 24, model_id: Optional[str] = None) -> Dict:
        """Get all images with optional filtering and pagination"""
        db = get_db()
        offset = (page - 1) * per_page

        # Build query based on filters
        query = """
            SELECT id, file_path, created_at, metadata
            FROM images
        """
        params = []

        if model_id:
            query += " WHERE json_extract(metadata, '$.model_id') = ?"
            params.append(model_id)

        # Get total count
        count_query = f"SELECT COUNT(*) as total FROM ({query})"
        total = db.execute(count_query, params).fetchone()["total"]

        # Add pagination
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([per_page, offset])

        # Get images
        images = db.execute(query, params).fetchall()

        return {
            "images": [{
                "id": img["id"],
                "created_at": img["created_at"],  # SQLite returns datetime object directly
                "metadata": json.loads(img["metadata"]),
                "model_id": json.loads(img["metadata"])["model_id"],
                "prompt": json.loads(img["metadata"])["prompt"]
            } for img in images],
            "total": total,
            "pages": (total + per_page - 1) // per_page,
            "current_page": page
        }

    @staticmethod
    def delete_image(image_id: str) -> bool:
        """Delete an image and its associated database record"""
        db = get_db()

        try:
            # Get image path
            result = db.execute(
                "SELECT file_path FROM images WHERE id = ?",
                (image_id,)
            ).fetchone()

            if result is None:
                return False

            # Delete the file
            file_path = os.path.join(current_app.config["IMAGES_PATH"], result["file_path"])
            if os.path.exists(file_path):
                os.remove(file_path)

            # Delete database record
            db.execute("DELETE FROM images WHERE id = ?", (image_id,))
            db.commit()

            return True

        except Exception as e:
            db.rollback()
            current_app.logger.error(f"Failed to delete image {image_id}: {str(e)}")
            return False

    @staticmethod
    def get_images_for_job(job_id: str) -> List[Dict]:
        """Get all images associated with a specific job"""
        db = get_db()
        images = db.execute(
            """
            SELECT id, file_path, created_at, metadata
            FROM images
            WHERE job_id = ?
            ORDER BY created_at DESC
            """,
            (job_id,)
        ).fetchall()

        return [{
            "id": img["id"],
            "file_path": img["file_path"],
            "created_at": ImageManager._convert_to_local(img["created_at"]),
            "metadata": json.loads(img["metadata"]),
            "model_id": json.loads(img["metadata"])["model_id"],
            "prompt": json.loads(img["metadata"])["prompt"]
        } for img in images]