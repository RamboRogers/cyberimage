"""
Model Context Protocol (MCP) implementation for CyberImage API
"""
import json
import logging
import uuid
import time
from flask import jsonify, request, current_app, Blueprint, g
from app.models import AVAILABLE_MODELS
from app.utils.queue import QueueManager
from app.api.routes import APIError

mcp_bp = Blueprint('mcp', __name__)

# Configure logger
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "flux-2"

@mcp_bp.route("/mcp", methods=["POST"])
def handle_mcp():
    """
    Handle MCP requests according to the protocol specification
    https://spec.modelcontextprotocol.io/specification/2024-11-05/
    """
    try:
        # Parse the MCP request
        mcp_request = request.get_json()

        # Validate basic JSON-RPC 2.0 structure
        if not mcp_request or "jsonrpc" not in mcp_request or mcp_request["jsonrpc"] != "2.0":
            return jsonify({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: Not a valid JSON-RPC 2.0 request"
                },
                "id": mcp_request.get("id", None)
            }), 400

        # Get method, params, and request ID
        method = mcp_request.get("method", "")
        params = mcp_request.get("params", {})
        req_id = mcp_request.get("id", str(uuid.uuid4()))

        # Route to appropriate handler based on method
        if method == "context.image_generation.generate":
            result = handle_generate(params)
            return jsonify({
                "jsonrpc": "2.0",
                "result": result,
                "id": req_id
            })
        elif method == "context.image_generation.status":
            result = handle_status(params)
            return jsonify({
                "jsonrpc": "2.0",
                "result": result,
                "id": req_id
            })
        elif method == "context.image_generation.models":
            result = handle_models(params)
            return jsonify({
                "jsonrpc": "2.0",
                "result": result,
                "id": req_id
            })
        else:
            # Method not found
            return jsonify({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                },
                "id": req_id
            }), 404

    except Exception as e:
        logger.error(f"MCP error: {str(e)}")
        return jsonify({
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": f"Server error: {str(e)}"
            },
            "id": request.get_json().get("id", None) if request.get_json() else None
        }), 500

def handle_generate(params):
    """
    Handle image generation requests

    Expected params:
    {
        "prompt": "A detailed description of the image",
        "negative_prompt": "What to avoid in the image (optional)",
        "model": "model-id (optional, defaults to flux-2)",
        "settings": {
            "height": 1024 (optional),
            "width": 1024 (optional),
            "num_inference_steps": 30 (optional),
            "guidance_scale": 7.5 (optional),
            "num_images": 1 (optional)
        }
    }
    """
    # Validate required parameters
    if "prompt" not in params:
        raise APIError("Missing required parameter: prompt", 400)

    # Extract parameters
    prompt = params["prompt"]
    negative_prompt = params.get("negative_prompt", "")
    model_id = params.get("model", DEFAULT_MODEL)

    # Validate model
    if model_id not in AVAILABLE_MODELS:
        available_models = ", ".join(AVAILABLE_MODELS.keys())
        raise APIError(f"Invalid model. Available models: {available_models}", 400)

    # Extract settings
    settings = params.get("settings", {})

    # Add negative prompt to settings if provided
    if negative_prompt:
        settings["negative_prompt"] = negative_prompt

    # Get number of images to generate (default to 1)
    num_images = min(int(settings.get("num_images", 1)), 8)
    settings["num_images"] = num_images

    # Create job in the database
    job_id = QueueManager.add_job(model_id, prompt, settings)

    # Add to generation queue
    g.generator.add_job({
        "id": job_id,
        "model_id": model_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "settings": settings
    })

    logger.info(f"MCP: Added job to queue: {job_id} for {num_images} images")

    return {
        "job_id": job_id,
        "status": "pending",
        "num_images": num_images
    }

def handle_status(params):
    """
    Handle job status requests

    Expected params:
    {
        "job_id": "UUID of the job"
    }
    """
    # Validate required parameters
    if "job_id" not in params:
        raise APIError("Missing required parameter: job_id", 400)

    job_id = params["job_id"]
    job_status = QueueManager.get_job(job_id)

    if not job_status:
        raise APIError(f"Job not found: {job_id}", 404)

    # Format response according to MCP standards
    response = {
        "job_id": job_id,
        "status": job_status["status"],
        "model": job_status["model_id"],
        "prompt": job_status["prompt"],
        "created_at": job_status.get("created_at", ""),
        "started_at": job_status.get("started_at", ""),
        "completed_at": job_status.get("completed_at", ""),
        "progress": job_status.get("progress", {}),
    }

    # Include images if job is completed
    if job_status["status"] == "completed" and "images" in job_status:
        response["images"] = [{
            "id": img["id"],
            "url": f"/api/get_image/{img['id']}",
            "metadata": img.get("metadata", {})
        } for img in job_status["images"]]

    return response

def handle_models(params):
    """
    Handle models listing requests
    """
    # Return available models with flux-2 marked as default
    return {
        "models": {name: {"id": info["id"], "description": info["description"]}
                 for name, info in AVAILABLE_MODELS.items()},
        "default": DEFAULT_MODEL
    }