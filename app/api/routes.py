"""
API Routes for CyberImage
"""
import os
import logging
import json
from flask import jsonify, request, send_file, current_app, g
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
from app.api import bp
from app.models import AVAILABLE_MODELS, DEFAULT_MODEL
from app.utils.queue import QueueManager
from app.utils.image import ImageManager
from app.utils.rate_limit import rate_limit
from app.utils.db import get_db
from openai import OpenAI
from app.models.generator import GenerationPipeline

# Configure logger
logger = logging.getLogger(__name__)

class APIError(Exception):
    """Base class for API errors"""
    def __init__(self, message, status_code=400, payload=None):
        super().__init__()
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        rv["status"] = "error"
        return rv

@bp.errorhandler(APIError)
def handle_api_error(error):
    """Handle API errors"""
    response = jsonify(error.to_dict())
    response.status_code = error.status_code

    # Add Retry-After header for rate limiting responses
    if error.status_code == 429 and error.payload and "retry_after" in error.payload:
        response.headers["Retry-After"] = str(error.payload["retry_after"])

    return response

@bp.route("/models", methods=["GET"])
def get_models():
    """Return list of available models"""
    try:
        return jsonify({
            "models": AVAILABLE_MODELS,
            "default": DEFAULT_MODEL
        })
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise APIError("Failed to get models", 500)

@bp.route("/generate", methods=["POST"])
@rate_limit
def generate_image():
    """Submit an image generation request"""
    try:
        # Check if generation is already in progress
        if GenerationPipeline._generation_in_progress:
            # If generation is in progress, return a 429 Too Many Requests
            queue_size = QueueManager.get_queue_status()["pending"]
            raise APIError(
                f"System is currently processing another request. Please try again later. Queue size: {queue_size}",
                429,
                {"retry_after": 30}  # Suggest client retry after 30 seconds
            )

        data = request.get_json()
        if not data:
            raise APIError("No data provided", 400)

        # Validate required fields
        required_fields = ["model_id", "prompt"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise APIError(f"Missing required fields: {', '.join(missing_fields)}", 400)

        # Validate model_id
        model_id = data["model_id"]
        if model_id not in AVAILABLE_MODELS:
            raise APIError(
                f"Invalid model_id. Available models: {', '.join(AVAILABLE_MODELS.keys())}",
                400
            )

        # No prompt length validation - let the LLM handle truncation
        # The frontend will display the character count but won't restrict submission

        # Validate negative prompt if provided (keeping this validation as negative prompts typically don't need to be as long)
        negative_prompt = data.get("negative_prompt", "")
        if negative_prompt and len(negative_prompt) > 2000:  # Increased limit for negative prompt too
            raise APIError("Negative prompt too long. Maximum length is 2000 characters.", 400)

        # Get optional settings
        settings = data.get("settings", {})

        # Add negative prompt to settings if provided
        if negative_prompt:
            settings["negative_prompt"] = negative_prompt

        # Get number of images to generate (default to 1)
        num_images = min(int(settings.get("num_images", 1)), 8)
        settings["num_images"] = num_images

        # Create job in the database first
        job_id = QueueManager.add_job(model_id, data["prompt"], settings)

        # Add to generation queue - just pass the job ID to trigger processing
        g.generator.add_job({
            "id": job_id,
            "model_id": model_id,
            "prompt": data["prompt"],
            "negative_prompt": negative_prompt,
            "settings": settings
        })

        logger.info(f"Added job to queue: {job_id} for {num_images} images")

        return jsonify({
            "job_id": job_id,
            "status": "pending",
            "num_images": num_images,
            "message": f"Image generation job submitted successfully for {num_images} image(s)"
        })

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error submitting generation job: {str(e)}")
        raise APIError("Failed to submit generation job", 500)

@bp.route("/status/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Get the status of a specific job"""
    try:
        status = QueueManager.get_job_status(job_id)
        if "error" in status:
            raise APIError(status["error"], 404)

        # If job is completed, include image information
        if status["status"] == "completed":
            # Check if we already have image_ids from the status
            if "image_ids" in status:
                # Use the stored image IDs
                image_list = []
                for image_id in status["image_ids"]:
                    try:
                        # Get each image info using the image manager
                        image_info = ImageManager.get_image_info(image_id)
                        if image_info:
                            image_list.append(image_info)
                    except Exception as e:
                        logger.warning(f"Error getting image info for {image_id}: {str(e)}")

                status["images"] = image_list
            else:
                # Fall back to the previous method of getting all job images
                images = ImageManager.get_job_images(job_id)
                status["images"] = images

        return jsonify(status)

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise APIError("Failed to get job status", 500)

@bp.route("/reset_queue", methods=["POST"])
def reset_queue():
    """Reset any stalled jobs to pending status"""
    try:
        reset_count = QueueManager.reset_stalled_jobs()

        # Trigger queue processing to pick up reset jobs
        if reset_count > 0 and hasattr(g, 'generator'):
            g.generator._queue_event.set()

        return jsonify({
            "status": "success",
            "reset_count": reset_count,
            "message": f"Reset {reset_count} stalled jobs to pending status"
        })
    except Exception as e:
        logger.error(f"Error resetting queue: {str(e)}")
        raise APIError("Failed to reset queue", 500)

@bp.route("/get_image/<image_id>", methods=["GET"])
def get_image(image_id):
    """Get a generated image"""
    try:
        image_path = ImageManager.get_image_path(image_id)
        if not image_path:
            raise APIError("Image not found", 404)

        return send_file(image_path, mimetype="image/png")

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving image: {str(e)}")
        raise APIError("Failed to retrieve image", 500)

@bp.route("/image/<image_id>/metadata", methods=["GET"])
def get_image_metadata(image_id):
    """Get image metadata"""
    try:
        metadata = ImageManager.get_image_metadata(image_id)
        if not metadata:
            raise APIError("Image not found", 404)

        return jsonify(metadata)

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metadata: {str(e)}")
        raise APIError("Failed to retrieve metadata", 500)

@bp.route("/image/<image_id>", methods=["DELETE"])
def delete_image(image_id):
    """Delete an image"""
    try:
        success = ImageManager.delete_image(image_id)
        if not success:
            raise APIError("Image not found", 404)

        return jsonify({
            "status": "success",
            "message": "Image deleted successfully"
        })

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        raise APIError("Failed to delete image", 500)

@bp.route("/queue", methods=["GET"])
def get_queue():
    """Return current queue status with enhanced statistics"""
    try:
        # Get basic queue status
        status = QueueManager.get_queue_status()

        # Add in-memory queue size if available
        try:
            if hasattr(g, 'generator') and g.generator and g.generator.generation_queue:
                status["queue_size"] = g.generator.generation_queue.qsize()
            else:
                status["queue_size"] = 0
        except Exception as e:
            current_app.logger.warning(f"Could not get queue size: {str(e)}")
            status["queue_size"] = 0

        # Get enhanced statistics if requested
        detailed = request.args.get('detailed', 'false').lower() == 'true'
        if detailed:
            # Get additional statistics from the database
            db = get_db()

            # Get recent job counts (last 24 hours)
            recent_stats = db.execute(
                """
                SELECT status, COUNT(*) as count
                FROM jobs
                WHERE created_at > datetime('now', '-1 day')
                GROUP BY status
                """
            ).fetchall()

            recent_counts = {
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "total": 0
            }

            for stat in recent_stats:
                recent_counts[stat["status"]] = stat["count"]
                recent_counts["total"] += stat["count"]

            status["recent_24h"] = recent_counts

            # Get average processing time for completed jobs
            avg_time = db.execute(
                """
                SELECT AVG(strftime('%s', completed_at) - strftime('%s', started_at)) as avg_time
                FROM jobs
                WHERE status = 'completed'
                AND completed_at IS NOT NULL
                AND started_at IS NOT NULL
                AND created_at > datetime('now', '-1 day')
                """
            ).fetchone()

            status["avg_processing_time_seconds"] = round(avg_time["avg_time"] or 0, 2)

            # Get model-specific statistics
            model_stats = db.execute(
                """
                SELECT model_id, status, COUNT(*) as count
                FROM jobs
                WHERE created_at > datetime('now', '-1 day')
                GROUP BY model_id, status
                """
            ).fetchall()

            models = {}
            for stat in model_stats:
                model_id = stat["model_id"]
                if model_id not in models:
                    models[model_id] = {
                        "pending": 0,
                        "processing": 0,
                        "completed": 0,
                        "failed": 0,
                        "total": 0
                    }

                models[model_id][stat["status"]] = stat["count"]
                models[model_id]["total"] += stat["count"]

            status["models"] = models

            # Get failure rate
            if status["completed"] + status["failed"] > 0:
                status["failure_rate"] = round(
                    (status["failed"] / (status["completed"] + status["failed"])) * 100, 2
                )
            else:
                status["failure_rate"] = 0

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        raise APIError("Failed to get queue status", 500)

@bp.route('/enrich', methods=['POST'])
def enrich_prompt():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400

        original_prompt = data['prompt']
        style = data.get('style', 'cyberpunk')  # Default to cyberpunk if not specified

        # Initialize OpenAI client
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_ENDPOINT')
        )

        # Get style-specific system prompt
        system_prompt = get_style_system_prompt(style)

        # Make API call to OpenAI
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4'),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Enhance this prompt: {original_prompt}"}
            ],
            temperature=0.7,
            max_tokens=200
        )

        enriched_prompt = response.choices[0].message.content.strip()

        return jsonify({
            'original_prompt': original_prompt,
            'enriched_prompt': enriched_prompt,
            'style': style
        })

    except Exception as e:
        current_app.logger.error(f"Error enriching prompt: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_style_system_prompt(style):
    """Get the appropriate system prompt based on the selected style."""
    style_prompts = {
        "cyberpunk": """You are a creative prompt engineer specializing in cyberpunk and sci-fi image generation prompts.
Your task is to enhance the given prompt by incorporating specific elements that work well with AI image generation.

Focus on adding these elements while maintaining the original intent:
- Cyberpunk elements (neon lights, high tech, low life, urban decay, corporate dystopia)
- Cinematic elements (dramatic lighting, depth of field, camera angles, composition)
- Technical details (8k, ultra detailed, octane render, unreal engine 5)
- Artistic style (rule of thirds, asymmetrical, strong negative space)
- Mood and atmosphere (time of day, weather, lighting conditions)
- Production quality (cinematic, professional photography, high production value)

Guidelines:
- Keep the original intent and core elements
- Make it more detailed and visually descriptive
- Use commas to separate concepts
- Keep it concise and focused
- Do not add any explanations or notes - return only the enhanced prompt
- Do not exceed 200 tokens

Example input: "a cyberpunk street"
Example output: "a neon-lit cyberpunk street at night, towering megacorporation buildings, steam rising from vents, holographic advertisements reflecting in puddles, cinematic composition, volumetric lighting, ultra detailed, 8k, octane render"
""",
        "anime": """You are a creative prompt engineer specializing in anime-style image generation prompts.
Your task is to enhance the given prompt by incorporating specific elements that work well with AI image generation.

Focus on adding these elements while maintaining the original intent:
- Anime art style elements (cel shading, vibrant colors, expressive features)
- Iconic anime aesthetics (big eyes, dramatic poses, exaggerated expressions)
- Studio references (Studio Ghibli, Kyoto Animation, Trigger, etc.)
- Technical details (8k, ultra detailed, anime key visual)
- Artistic style (dynamic composition, vibrant color palette, clean linework)
- Mood and atmosphere (lighting effects, background details, emotional tone)

Guidelines:
- Keep the original intent and core elements
- Make it more detailed and visually descriptive
- Use commas to separate concepts
- Keep it concise and focused
- Do not add any explanations or notes - return only the enhanced prompt
- Do not exceed 200 tokens

Example input: "a girl in a forest"
Example output: "a young anime girl with flowing hair in a mystical forest, dappled sunlight, Studio Ghibli style, vibrant colors, detailed background, expressive big eyes, cel shading, magical atmosphere, key visual, 8k, ultra detailed"
""",
        "realistic": """You are a creative prompt engineer specializing in realistic and photorealistic image generation prompts.
Your task is to enhance the given prompt by incorporating specific elements that work well with AI image generation.

Focus on adding these elements while maintaining the original intent:
- Photorealistic details (texture, lighting, imperfections, materials)
- Photography elements (lens type, focal length, depth of field, exposure)
- Technical specifications (4k, 8k, RAW, ultra detailed, professional photography)
- Lighting conditions (golden hour, studio lighting, natural light, rim light)
- Composition elements (rule of thirds, framing, perspective, bokeh)
- Environmental details (location specifics, weather, time of day)

Guidelines:
- Keep the original intent and core elements
- Make it more detailed and visually descriptive
- Use commas to separate concepts
- Keep it concise and focused
- Do not add any explanations or notes - return only the enhanced prompt
- Do not exceed 200 tokens

Example input: "a portrait of a man"
Example output: "portrait of a man with subtle skin texture and pores, professional studio lighting, soft shadows, shallow depth of field, shot on Canon EOS R5, 85mm lens f/1.2, natural expression, detailed eyes, professional color grading, 8k, photorealistic, award-winning photography"
""",
        "enhance": """You are a creative prompt engineer specializing in enhancing image generation prompts.
Your task is to enhance the given prompt by incorporating specific elements that work well with AI image generation, while maintaining the original style and intent.

Focus on adding these elements while maintaining the original intent:
- Technical details (8k, ultra detailed, professional lighting)
- Artistic style (composition, color palette, visual interest)
- Mood and atmosphere (lighting, weather, time of day)
- Production quality (cinematic, professional photography, high production value)
- Detailed elements (textures, materials, environmental details)

Guidelines:
- Keep the original intent and core elements
- Make it more detailed and visually descriptive
- Use commas to separate concepts
- Keep it concise and focused
- Do not add any explanations or notes - return only the enhanced prompt
- Do not exceed 200 tokens

Example input: "a castle on a hill"
Example output: "a majestic castle on a rolling hill, dramatic sky with scattered clouds, golden hour lighting, detailed stone textures, lush surrounding landscape, cinematic composition, professional photography, atmospheric, highly detailed, 8k resolution"
""",
        "fantasy": """You are a creative prompt engineer specializing in fantasy and magical image generation prompts.
Your task is to enhance the given prompt by incorporating specific elements that work well with AI image generation.

Focus on adding these elements while maintaining the original intent:
- Fantasy elements (magical creatures, enchanted objects, mystical environments)
- Aesthetic details (otherworldly lighting, ethereal atmosphere, magical effects)
- Technical quality (8k, ultra detailed, unreal engine 5, digital painting)
- Artistic style (dramatic composition, fantasy color palette, dynamic lighting)
- World-building details (magical ecosystems, fantasy architecture, ancient runes)
- Production quality (concept art, professional illustration, digital masterpiece)

Guidelines:
- Keep the original intent and core elements
- Make it more detailed and visually descriptive
- Use commas to separate concepts
- Keep it concise and focused
- Do not add any explanations or notes - return only the enhanced prompt
- Do not exceed 200 tokens

Example input: "a wizard casting a spell"
Example output: "a powerful wizard with ornate robes casting an arcane spell, magical energy swirling around hands, glowing runes floating in air, enchanted staff with crystals, ancient library backdrop, magical tomes, ethereal lighting, mystical atmosphere, digital painting, concept art, ultra detailed, 8k resolution"
""",
        "sci-fi": """You are a creative prompt engineer specializing in science fiction image generation prompts.
Your task is to enhance the given prompt by incorporating specific elements that work well with AI image generation.

Focus on adding these elements while maintaining the original intent:
- Sci-fi elements (advanced technology, futuristic architecture, alien worlds)
- Technical futurism (holographic interfaces, energy fields, quantum mechanics)
- Technical details (8k, ultra detailed, octane render, unreal engine 5)
- Artistic style (futuristic composition, sci-fi color palette, dynamic lighting)
- Environment details (alien atmosphere, space phenomena, futuristic cityscapes)
- Production quality (concept art, cinematic, professional sci-fi illustration)

Guidelines:
- Keep the original intent and core elements
- Make it more detailed and visually descriptive
- Use commas to separate concepts
- Keep it concise and focused
- Do not add any explanations or notes - return only the enhanced prompt
- Do not exceed 200 tokens

Example input: "a spaceship"
Example output: "advanced interstellar spaceship with sleek aerodynamic design, glowing thrusters, exotic alloy hull with intricate panel details, holographic displays, quantum drive core, orbiting a distant exoplanet with twin moons, cosmic nebula in background, cinematic lighting, concept art, hyper-realistic, 8k, ultra detailed, unreal engine 5 render"
"""
    }

    # Return the requested style prompt or the default enhance prompt if not found
    return style_prompts.get(style, style_prompts["enhance"])

@bp.route('/gallery', methods=['GET'])
def get_gallery():
    """Return paginated gallery images with search capabilities"""
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        search_query = request.args.get('search', None)
        model_filter = request.args.get('model', None)

        # Cap the limit to avoid extremely large queries
        if limit > 50:
            limit = 50

        # Calculate offset for pagination (when not searching)
        offset = (page - 1) * limit

        db = get_db()

        # Build search conditions for the images table
        conditions = ["1=1"]  # Default condition that's always true
        params = []

        # Add search filter if provided - support multi-word search
        if search_query and search_query.strip():
            # Split the search query into keywords and create conditions for each
            keywords = search_query.strip().split()

            keyword_conditions = []
            for keyword in keywords:
                term_conditions = []
                # Search in prompt
                term_conditions.append("json_extract(metadata, '$.prompt') LIKE ?")
                params.append(f"%{keyword}%")

                # Search in model_id
                term_conditions.append("json_extract(metadata, '$.model_id') LIKE ?")
                params.append(f"%{keyword}%")

                # Allow matching either prompt OR model_id for each keyword
                keyword_conditions.append(f"({' OR '.join(term_conditions)})")

            # Join the keyword conditions with AND (all keywords must match)
            if keyword_conditions:
                conditions.append(f"({' AND '.join(keyword_conditions)})")

        # Add specific model filter if provided
        if model_filter:
            conditions.append("json_extract(metadata, '$.model_id') = ?")
            params.append(model_filter)

        # Build the where clause
        where_clause = " AND ".join(conditions)

        # Get total count of images matching conditions
        count_query = f"""
            SELECT COUNT(*) as count
            FROM images
            WHERE {where_clause}
        """

        total_count = db.execute(count_query, params).fetchone()["count"]

        # For search queries, don't apply pagination - return all results
        # For browsing (no search), apply normal pagination
        if search_query and search_query.strip():
            # When searching, don't limit the results - return ALL matching items
            query = f"""
                SELECT
                    id,
                    json_extract(metadata, '$.model_id') as model_id,
                    json_extract(metadata, '$.prompt') as prompt,
                    created_at,
                    metadata as settings
                FROM images
                WHERE {where_clause}
                ORDER BY created_at DESC
            """
            query_params = params.copy()
            logger.info(f"Searching without pagination, found {total_count} matches for '{search_query}'")
        else:
            # Normal browsing mode - apply pagination
            query = f"""
                SELECT
                    id,
                    json_extract(metadata, '$.model_id') as model_id,
                    json_extract(metadata, '$.prompt') as prompt,
                    created_at,
                    metadata as settings
                FROM images
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            # Add pagination parameters
            query_params = params.copy()
            query_params.extend([limit, offset])

        # Execute the query
        images = db.execute(query, query_params).fetchall()

        has_more = False if search_query else total_count > (offset + limit)

        formatted_images = []
        for image in images:
            # Extract settings - should be a JSON string
            settings = image["settings"]
            if isinstance(settings, str):
                try:
                    settings = json.loads(settings)
                except:
                    settings = {}

            formatted_images.append({
                "id": image["id"],
                "model_id": image["model_id"],
                "prompt": image["prompt"],
                "created_at": image["created_at"],
                "settings": settings
            })

        return jsonify({
            "images": formatted_images,
            "page": page,
            "limit": limit,
            "has_more": has_more,
            "total": total_count
        })

    except Exception as e:
        logger.error(f"Error fetching gallery images: {str(e)}")
        logger.error(f"Exception details: {type(e).__name__}, {str(e)}")
        raise APIError("Failed to fetch gallery images", 500)