"""
API Routes for CyberImage
"""
import os
import logging
from flask import jsonify, request, send_file, current_app, g
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
from app.api import bp
from app.models import AVAILABLE_MODELS
from app.utils.queue import QueueManager
from app.utils.image import ImageManager
from app.utils.rate_limit import rate_limit
from openai import OpenAI

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
    return response

@bp.route("/models", methods=["GET"])
def get_models():
    """Return list of available models"""
    try:
        return jsonify({
            "models": AVAILABLE_MODELS,
            "default": "flux-2"
        })
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        raise APIError("Failed to get models", 500)

@bp.route("/generate", methods=["POST"])
@rate_limit
def generate_image():
    """Submit an image generation request"""
    try:
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

        # Validate prompt length
        if len(data["prompt"]) > 500:
            raise APIError("Prompt too long. Maximum length is 500 characters.", 400)

        # Validate negative prompt if provided
        negative_prompt = data.get("negative_prompt", "")
        if negative_prompt and len(negative_prompt) > 500:
            raise APIError("Negative prompt too long. Maximum length is 500 characters.", 400)

        # Get optional settings
        settings = data.get("settings", {})

        # Add negative prompt to settings if provided
        if negative_prompt:
            settings["negative_prompt"] = negative_prompt

        # Get number of images to generate (default to 1)
        num_images = min(int(settings.get("num_images", 1)), 8)
        settings["num_images"] = num_images

        # Create job
        job_id = QueueManager.add_job(model_id, data["prompt"], settings)

        # Add to generation queue
        g.generator.add_job({
            "id": job_id,
            "model_id": model_id,
            "prompt": data["prompt"],
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
            images = ImageManager.get_job_images(job_id)
            status["images"] = images

        return jsonify(status)

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise APIError("Failed to get job status", 500)

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
    """Return current queue status"""
    try:
        status = QueueManager.get_queue_status()
        # Safely get queue size from generator if available
        try:
            if hasattr(g, 'generator') and g.generator and g.generator.generation_queue:
                status["queue_size"] = g.generator.generation_queue.qsize()
            else:
                status["queue_size"] = 0
        except Exception as e:
            current_app.logger.warning(f"Could not get queue size: {str(e)}")
            status["queue_size"] = 0
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

        # Initialize OpenAI client
        client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_ENDPOINT')
        )

        # System prompt for enrichment
        system_prompt = """You are a creative prompt engineer specializing in cyberpunk and sci-fi image generation prompts.
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
"""

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
            'enriched_prompt': enriched_prompt
        })

    except Exception as e:
        current_app.logger.error(f"Error enriching prompt: {str(e)}")
        return jsonify({'error': str(e)}), 500