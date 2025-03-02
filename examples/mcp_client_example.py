"""
Example of how to use the CyberImage MCP endpoint
This demonstrates the Model Context Protocol implementation for AI image generation
"""
import json
import requests
import time
import uuid
import os
from PIL import Image
import io
import base64

# API configuration
API_URL = "http://localhost:5050/api/mcp"  # Adjust for your deployment

def mcp_request(method, params=None):
    """
    Make an MCP request to the CyberImage API

    Args:
        method (str): The MCP method to call
        params (dict): Parameters for the method

    Returns:
        dict: The API response
    """
    request_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": request_id
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    return response.json()

def list_models():
    """Get available models"""
    response = mcp_request("context.image_generation.models")

    if not response or "error" in response:
        print("Error listing models:", response.get("error", {}).get("message", "Unknown error"))
        return

    result = response["result"]
    print("\nAvailable Models:")
    print("================")

    for name, info in result["models"].items():
        print(f"- {name}: {info['description']}")

    print(f"\nDefault model: {result['default']}")
    return result

def generate_image(prompt, model=None, negative_prompt="", settings=None):
    """
    Generate an image using the CyberImage MCP API

    Args:
        prompt (str): The image description
        model (str, optional): Model to use
        negative_prompt (str, optional): What to exclude from the image
        settings (dict, optional): Additional generation settings

    Returns:
        str: Job ID for the generation request
    """
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt
    }

    if model:
        params["model"] = model

    if settings:
        params["settings"] = settings

    print(f"Generating image with prompt: '{prompt}'")
    print(f"Using model: {model or 'default'}")

    response = mcp_request("context.image_generation.generate", params)

    if not response or "error" in response:
        print("Error generating image:", response.get("error", {}).get("message", "Unknown error"))
        return None

    result = response["result"]
    job_id = result["job_id"]
    print(f"Job submitted with ID: {job_id}")
    print(f"Status: {result['status']}")

    return job_id

def check_status(job_id):
    """
    Check the status of a generation job

    Args:
        job_id (str): The job ID to check

    Returns:
        dict: The job status
    """
    response = mcp_request("context.image_generation.status", {"job_id": job_id})

    if not response or "error" in response:
        print("Error checking status:", response.get("error", {}).get("message", "Unknown error"))
        return None

    return response["result"]

def download_image(image_url, save_path):
    """
    Download an image from the given URL

    Args:
        image_url (str): The image URL
        save_path (str): Where to save the image

    Returns:
        bool: True if successful, False otherwise
    """
    # Get full URL
    if image_url.startswith("/"):
        image_url = f"http://localhost:5050{image_url}"

    response = requests.get(image_url)

    if response.status_code != 200:
        print(f"Error downloading image: {response.status_code}")
        return False

    with open(save_path, "wb") as f:
        f.write(response.content)

    print(f"Image saved to {save_path}")
    return True

def wait_for_completion(job_id, poll_interval=5, max_attempts=60):
    """
    Wait for a job to complete

    Args:
        job_id (str): The job ID to check
        poll_interval (int): How often to check the status in seconds
        max_attempts (int): Maximum number of polling attempts

    Returns:
        dict: Final job status or None if timed out
    """
    attempts = 0

    while attempts < max_attempts:
        status = check_status(job_id)

        if not status:
            return None

        print(f"Status: {status['status']}")

        if status["status"] == "processing" and "progress" in status:
            progress = status["progress"]
            if progress.get("generating") and progress.get("step") is not None:
                print(f"Generating: Step {progress['step']}/{progress['total_steps']}")

        if status["status"] in ["completed", "failed"]:
            return status

        time.sleep(poll_interval)
        attempts += 1

    print(f"Timed out after {max_attempts * poll_interval} seconds")
    return None

def main():
    """Main function demonstrating the MCP client"""
    # List available models
    models_info = list_models()

    if not models_info:
        return

    # Use the default model
    default_model = models_info["default"]

    # Generate an image
    prompt = "A futuristic cyberpunk city at night with neon lights and flying cars"
    negative_prompt = "blurry, low quality, distorted"

    settings = {
        "num_images": 1,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "height": 1024,
        "width": 1024
    }

    job_id = generate_image(
        prompt=prompt,
        model=default_model,
        negative_prompt=negative_prompt,
        settings=settings
    )

    if not job_id:
        return

    # Wait for the job to complete
    final_status = wait_for_completion(job_id)

    if not final_status:
        return

    if final_status["status"] == "completed" and "images" in final_status:
        # Download each generated image
        for i, image_info in enumerate(final_status["images"]):
            image_url = image_info["url"]
            save_path = f"generated_image_{i+1}.png"
            download_image(image_url, save_path)
    elif final_status["status"] == "failed":
        print(f"Generation failed: {final_status.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()