"""
Test script for CyberImage generation pipeline
"""
import sys
import time
import requests
from typing import Dict, Optional
from app.models import AVAILABLE_MODELS

# Configuration
API_BASE = "http://localhost:5050/api"
TEST_PROMPTS = {
    "flux-2": "A cyberpunk city at night with neon signs and flying cars, highly detailed",
    "flux-1": "A futuristic metropolis with towering skyscrapers and holographic billboards",
    "sd-3.5": "A futuristic robot in a zen garden, detailed digital art",
    "flux-abliterated": "A mystical forest with glowing mushrooms and floating crystals, fantasy art"
}

def print_status(message: str, status: str = "info") -> None:
    """Print formatted status messages"""
    status_icons = {
        "info": "ℹ️",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "pending": "⏳"
    }
    icon = status_icons.get(status, "ℹ️")
    print(f"\n{icon} {message}")
    sys.stdout.flush()

def submit_generation(model_id: str, prompt: str) -> Optional[str]:
    """Submit a generation request"""
    try:
        print_status(f"Testing model: {model_id}", "info")
        print_status(f"Prompt: {prompt}", "info")

        # Submit generation request
        print_status("Submitting generation request...", "pending")
        response = requests.post(f"{API_BASE}/generate", json={
            "model_id": model_id,
            "prompt": prompt,
            "settings": {
                "num_inference_steps": 35,
                "guidance_scale": 8.0,
                "height": 1024,
                "width": 1024,
                "max_sequence_length": 512
            }
        })
        response.raise_for_status()
        job_id = response.json()["job_id"]
        print_status(f"Job submitted successfully. Job ID: {job_id}", "success")
        return job_id
    except Exception as e:
        print_status(f"Failed to submit job: {str(e)}", "error")
        return None

def check_status(job_id: str, timeout: int = 300) -> bool:
    """Check job status with timeout"""
    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            print_status("Checking job status...", "pending")
            response = requests.get(f"{API_BASE}/status/{job_id}")
            response.raise_for_status()
            status = response.json()

            if status["status"] == "completed":
                print_status("Generation completed successfully!", "success")
                return True
            elif status["status"] == "failed":
                print_status(f"Generation failed: {status.get('error', 'Unknown error')}", "error")
                return False
            elif status["status"] in ["pending", "processing"]:
                print_status(f"Status: {status['status']}", "info")
                time.sleep(5)
                continue
            else:
                print_status(f"Unknown status: {status['status']}", "warning")
                return False

        print_status(f"Timeout after {timeout} seconds", "error")
        return False
    except Exception as e:
        print_status(f"Error checking status: {str(e)}", "error")
        return False

def test_model(model_id: str, prompt: str) -> Dict:
    """Test a single model"""
    results = {
        "model_id": model_id,
        "prompt": prompt,
        "success": False,
        "error": None
    }

    try:
        # Submit job
        job_id = submit_generation(model_id, prompt)
        if not job_id:
            results["error"] = "Failed to submit job"
            return results

        # Check status
        success = check_status(job_id)
        results["success"] = success
        if not success:
            results["error"] = "Generation failed or timed out"

    except Exception as e:
        results["error"] = str(e)

    return results

def main():
    """Main test function"""
    print_status("Starting CyberImage Generation Tests", "info")

    results = []
    for model_id in AVAILABLE_MODELS:
        prompt = TEST_PROMPTS.get(model_id, "A beautiful landscape at sunset, digital art")
        print_status(f"\nTesting {model_id}", "info")
        result = test_model(model_id, prompt)
        results.append(result)

    # Print summary
    print_status("\nTest Summary:", "info")
    for result in results:
        status = "success" if result["success"] else "error"
        message = f"Model: {result['model_id']} - {'Success' if result['success'] else 'Failed'}"
        if result["error"]:
            message += f" - Error: {result['error']}"
        print_status(message, status)

if __name__ == "__main__":
    main()