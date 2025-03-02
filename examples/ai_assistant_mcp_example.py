"""
Example of how an AI Assistant can use the CyberImage MCP endpoint for image generation
This demonstrates the Model Context Protocol implementation specifically for AI assistants
"""
import json
import requests
import time
import uuid
import os
import base64

# API configuration
API_URL = "http://localhost:5050/api/mcp"  # Adjust for your deployment

def generate_image_for_assistant(prompt, model=None, negative_prompt=""):
    """
    Complete function for AI assistants to generate an image and get a URL
    This function handles the entire process from generation request to downloading

    Args:
        prompt (str): The image description
        model (str, optional): Model ID to use (defaults to flux-2)
        negative_prompt (str, optional): What to exclude from the image

    Returns:
        dict: A dictionary with status and either the image URL or error message
    """
    # Step 1: Check available models
    models_info = list_models()
    if not models_info:
        return {"status": "error", "message": "Failed to retrieve model information"}

    # Use provided model or default
    model_id = model if model and model in models_info["models"] else models_info["default"]

    # Step 2: Generate image
    job_id = submit_generation_job(prompt, model_id, negative_prompt)
    if not job_id:
        return {"status": "error", "message": "Failed to submit generation job"}

    # Step 3: Wait for completion
    status = wait_for_job_completion(job_id)
    if not status:
        return {"status": "error", "message": "Job timed out or failed"}

    if status["status"] == "failed":
        return {"status": "error", "message": f"Generation failed: {status.get('message', 'Unknown error')}"}

    # Step 4: Return image information
    if status["status"] == "completed" and "images" in status and len(status["images"]) > 0:
        image_url = status["images"][0]["url"]
        # Convert relative URL to absolute URL
        if image_url.startswith("/"):
            image_url = f"http://localhost:5050{image_url}"

        return {
            "status": "success",
            "image_url": image_url,
            "prompt": prompt,
            "model": model_id
        }

    return {"status": "error", "message": "Unknown error occurred"}

def mcp_request(method, params=None):
    """
    Make an MCP request to the CyberImage API

    Args:
        method (str): The MCP method to call
        params (dict): Parameters for the method

    Returns:
        dict: The API response result or None on error
    """
    request_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": request_id
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        if "error" in result:
            print(f"API error: {result['error']['message']}")
            return None

        return result.get("result")
    except Exception as e:
        print(f"Request error: {str(e)}")
        return None

def list_models():
    """Get available models"""
    return mcp_request("context.image_generation.models")

def submit_generation_job(prompt, model, negative_prompt=""):
    """Submit a generation job and return the job ID"""
    params = {
        "prompt": prompt,
        "model": model,
        "negative_prompt": negative_prompt,
        "settings": {
            "num_images": 1,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024
        }
    }

    result = mcp_request("context.image_generation.generate", params)
    return result["job_id"] if result and "job_id" in result else None

def check_job_status(job_id):
    """Check status of a generation job"""
    return mcp_request("context.image_generation.status", {"job_id": job_id})

def wait_for_job_completion(job_id, max_wait_seconds=300, poll_interval=5):
    """Wait for a job to complete with timeout"""
    start_time = time.time()

    while time.time() - start_time < max_wait_seconds:
        status = check_job_status(job_id)

        if not status:
            return None

        if status["status"] in ["completed", "failed"]:
            return status

        time.sleep(poll_interval)

    return None

# Example usage
if __name__ == "__main__":
    # Example of how an AI assistant would call this function
    result = generate_image_for_assistant(
        prompt="A beautiful mountain landscape with a lake and forest at sunset",
        negative_prompt="blurry, distorted, low quality"
    )

    if result["status"] == "success":
        print(f"Image generated successfully!")
        print(f"Image URL: {result['image_url']}")
        print(f"Prompt: {result['prompt']}")
        print(f"Model: {result['model']}")
    else:
        print(f"Error: {result['message']}")