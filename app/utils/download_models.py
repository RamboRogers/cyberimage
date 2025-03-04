"""
Model downloader for CyberImage
"""
import os
import sys
import subprocess
from pathlib import Path
import shutil
import fcntl
from flask import current_app

# Model configurations based on actual repo structure
MODELS = {
    "flux-1": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "description": "FLUX base model",
        "requires_auth": True,
        "source": "huggingface"
    },
    "sd-3.5": {
        "repo": "stabilityai/stable-diffusion-3.5-large",
        "description": "Stable Diffusion 3.5",
        "requires_auth": True,
        "source": "huggingface"
    },
    "flux-abliterated": {
        "repo": "aoxo/flux.1dev-abliteratedv2",
        "description": "FLUX Abliterated variant",
        "requires_auth": True,
        "source": "huggingface"
    }
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

def download_model(models_dir: Path, model_name: str, model_info: dict) -> bool:
    """Download a complete model folder using HuggingFace CLI"""
    try:
        print_status(f"Downloading {model_name}: {model_info['description']}", "pending")

        model_path = models_dir / model_name
        temp_path = models_dir / f"{model_name}_temp"

        # Clean up any leftover temp directory
        if temp_path.exists():
            shutil.rmtree(temp_path)

        # Create temp directory
        temp_path.mkdir(exist_ok=True)
        env = os.environ.copy()

        # Disable HF_TRANSFER to avoid download errors
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

        # For HuggingFace models - download the entire folder
        if model_info["source"] == "huggingface":
            # Simple approach: download everything at once
            cmd = [
                "huggingface-cli", "download",
                model_info["repo"],
                "--local-dir", str(temp_path),
                "--local-dir-use-symlinks", "False"
            ]

            print_status(f"Downloading complete model: {model_name}...", "info")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print_status(f"Failed to download model: {result.stderr}", "error")
                if temp_path.exists():
                    shutil.rmtree(temp_path)
                return False

            # Move to final location
            if model_path.exists():
                shutil.rmtree(model_path)
            temp_path.rename(model_path)
            print_status(f"Successfully downloaded {model_name}", "success")
            return True

        # Keep Civitai download logic as is
        elif model_info["source"] == "civitai":
            # Extract model ID from repo string
            model_id = model_info["repo"].split(":")[1]

            # Get Civitai API key from environment
            api_key = os.getenv("CIVITAI_API_KEY")
            if not api_key:
                print_status("CIVITAI_API_KEY not found in environment", "error")
                return False

            # Download from Civitai
            import requests
            headers = {"Authorization": f"Bearer {api_key}"}

            # First get the model info to get the download URL
            info_url = f"https://civitai.com/api/v1/models/{model_id}"
            response = requests.get(info_url, headers=headers)
            if not response.ok:
                print_status(f"Failed to get model info from Civitai: {response.text}", "error")
                return False

            model_data = response.json()
            download_url = model_data["modelVersions"][0]["downloadUrl"]

            # Download the model file
            print_status(f"Downloading model from Civitai...", "info")
            response = requests.get(download_url, headers=headers, stream=True)
            if not response.ok:
                print_status(f"Failed to download from Civitai: {response.text}", "error")
                return False

            # Save the file
            model_file = temp_path / "model.safetensors"
            with open(model_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Move to final location
            if model_path.exists():
                shutil.rmtree(model_path)
            temp_path.rename(model_path)
            print_status(f"Successfully downloaded {model_name}", "success")
            return True

    except Exception as e:
        print_status(f"Failed to download {model_name}: {str(e)}", "error")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        return False

def acquire_lock(lock_path):
    """Try to acquire a lock file"""
    try:
        lock_file = open(lock_path, 'w')
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_file
    except (IOError, OSError):
        return None

def release_lock(lock_file):
    """Release the lock file"""
    if lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

def download_all_models():
    """Download all required models"""
    models_dir = Path(os.getenv("MODEL_FOLDER", "./models"))
    models_dir.mkdir(exist_ok=True)

    # Try to acquire lock
    lock_path = models_dir / ".download.lock"
    lock_file = acquire_lock(lock_path)

    if not lock_file:
        print_status("Another download process is already running", "info")
        return

    try:
        print_status("Checking existing models...", "info")

        # Check which models need to be downloaded
        models_to_download = {}
        for model_name, model_info in MODELS.items():
            model_path = models_dir / model_name

            # Simply check if the model directory exists
            if model_path.exists():
                print_status(f"Model {model_name} already exists, skipping...", "info")
                continue

            # If we get here, the model needs to be downloaded
            print_status(f"Model {model_name} is missing", "warning")
            models_to_download[model_name] = model_info

        if not models_to_download:
            print_status("All models are already downloaded!", "success")
            return

        print_status(f"Need to download {len(models_to_download)} models...", "info")

        # Download missing models
        success_count = 0
        for model_name, model_info in models_to_download.items():
            if download_model(models_dir, model_name, model_info):
                success_count += 1

        # Print summary
        total_models = len(models_to_download)
        print_status("\nDownload Summary:", "info")
        print(f"Successfully downloaded {success_count} of {total_models} models")

        if success_count == total_models:
            print_status("All models downloaded successfully!", "success")
        else:
            print_status(f"Some models failed to download ({total_models - success_count} failed)", "warning")
            sys.exit(1)  # Exit if models failed to download

    finally:
        # Always release the lock
        release_lock(lock_file)
        if lock_path.exists():
            lock_path.unlink()