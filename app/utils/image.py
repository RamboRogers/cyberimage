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
    def save_image(image: Image.Image, job_id: str, metadata: Dict) -> str:
        """Save an image and record it in the database"""
        db = get_db()
        image_id = str(uuid.uuid4())

        # Get job information for metadata
        job = db.execute(
            """
            SELECT model_id, prompt, negative_prompt, settings
            FROM jobs WHERE id = ?
            """,
            (job_id,)
        ).fetchone()

        if job:
            # Update metadata with job information
            metadata.update({
                "model_id": job["model_id"],
                "prompt": job["prompt"],
                "negative_prompt": job["negative_prompt"] if job["negative_prompt"] else None,
                "settings": json.loads(job["settings"])
            })

        # Create a directory structure by date to organize images
        today = datetime.utcnow().strftime("%Y/%m/%d")
        image_dir = os.path.join(current_app.config["IMAGES_PATH"], today)
        os.makedirs(image_dir, exist_ok=True)

        # Save the image
        file_name = f"{image_id}.png"
        file_path = os.path.join(image_dir, file_name)
        image.save(file_path, "PNG")

        # Record in database
        try:
            db.execute(
                """
                INSERT INTO images (id, job_id, file_path, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (image_id, job_id, os.path.join(today, file_name), json.dumps(metadata))
            )
            db.commit()
            return image_id
        except Exception as e:
            # Clean up the image file if database insert fails
            if os.path.exists(file_path):
                os.remove(file_path)
            db.rollback()
            raise Exception(f"Failed to save image: {str(e)}")

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