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
from app.utils.config import get_downloadable_models

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
        # Ensure model_name doesn't have quotes or other characters that might cause issues
        sanitized_name = model_name.strip('"\' \t')
        model_path = models_dir / sanitized_name
        temp_path = models_dir / f"{sanitized_name}_temp"

        print_status(f"Downloading {sanitized_name}: {model_info['description']}", "pending")

        # Clean up any leftover temp directory
        if temp_path.exists():
            print_status(f"Cleaning up existing temp directory: {temp_path}", "info")
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

            print_status(f"Downloading complete model: {sanitized_name}...", "info")
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                print_status(f"Failed to download model: {result.stderr}", "error")
                if temp_path.exists():
                    shutil.rmtree(temp_path)
                return False

            # Move to final location
            if model_path.exists():
                print_status(f"Removing existing model directory: {model_path}", "info")
                shutil.rmtree(model_path)

            print_status(f"Moving from {temp_path} to {model_path}", "info")
            temp_path.rename(model_path)
            print_status(f"Successfully downloaded {sanitized_name}", "success")
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
                print_status(f"Removing existing model directory: {model_path}", "info")
                shutil.rmtree(model_path)

            print_status(f"Moving from {temp_path} to {model_path}", "info")
            temp_path.rename(model_path)
            print_status(f"Successfully downloaded {sanitized_name}", "success")
            return True

    except Exception as e:
        print_status(f"Failed to download {sanitized_name}: {str(e)}", "error")
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

def clean_temp_directories(models_dir: Path):
    """Find and remove any temporary download directories"""
    print_status("Cleaning up any leftover temporary directories", "info")
    count = 0

    # Look for directories ending with _temp or containing quotes
    for path in models_dir.iterdir():
        if path.is_dir() and (path.name.endswith('_temp') or '"' in path.name or "'" in path.name):
            try:
                print_status(f"Removing temporary directory: {path.name}", "info")
                shutil.rmtree(path)
                count += 1
            except Exception as e:
                print_status(f"Failed to remove {path.name}: {str(e)}", "warning")

    if count > 0:
        print_status(f"Removed {count} temporary directories", "success")
    else:
        print_status("No temporary directories found", "info")

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
        # First clean up any leftover temp directories or directories with quotes
        clean_temp_directories(models_dir)

        print_status("Checking existing models...", "info")

        # Get models that are enabled for download from config
        models_to_download = {}
        enabled_models = get_downloadable_models()

        print_status(f"Found {len(enabled_models)} enabled models in configuration", "info")

        # Check which models need to be downloaded
        for model_name, model_info in enabled_models.items():
            # Sanitize model name
            sanitized_name = model_name.strip('"\' \t')
            model_path = models_dir / sanitized_name

            # Check if model directory exists and has all required files
            if model_path.exists() and check_model_files(model_path, model_info):
                print_status(f"Model {sanitized_name} already exists and is complete, skipping...", "info")
                continue

            # If we get here, the model needs to be downloaded
            if not model_path.exists():
                print_status(f"Model {sanitized_name} is missing", "warning")
            else:
                print_status(f"Model {sanitized_name} exists but is incomplete", "warning")

            models_to_download[model_name] = model_info

        if not models_to_download:
            print_status("All enabled models are already downloaded!", "success")
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

def check_model_files(model_path: Path, model_info: dict) -> bool:
    """Check if all required files exist for a model"""
    if not model_path.exists():
        return False

    # If no files are specified, just check if the directory exists
    if "files" not in model_info or not model_info["files"]:
        return True

    # Check each required file
    for file in model_info["files"]:
        file_path = model_path / file
        if not file_path.exists():
            print_status(f"Missing required file in {model_path.name}: {file}", "warning")
            return False

    return True