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

def print_model_structure(model_path: Path, max_depth: int = 3):
    """Print the directory structure of a model for debugging"""
    if not model_path.exists():
        print_status(f"Model path {model_path} does not exist", "error")
        return

    print_status(f"Model structure for {model_path.name}:", "info")

    def print_dir(path, prefix="", depth=0):
        if depth > max_depth:
            print(f"{prefix}... (max depth reached)")
            return

        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"{prefix}{path.name} ({size_mb:.2f} MB)")
        else:
            print(f"{prefix}{path.name}/")
            items = list(path.iterdir())
            items.sort(key=lambda x: (x.is_file(), x.name))

            for i, item in enumerate(items):
                is_last = (i == len(items) - 1)
                if is_last:
                    new_prefix = prefix + "└── "
                    child_prefix = prefix + "    "
                else:
                    new_prefix = prefix + "├── "
                    child_prefix = prefix + "│   "

                print_dir(item, new_prefix, depth + 1)

    # Start recursive printing
    print_dir(model_path)
    print()  # Add a blank line at the end

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

        # SAFETY CHECK: If the final model folder already exists, don't download again
        if model_path.exists():
            print_status(f"Model {sanitized_name} already exists, won't overwrite it", "warning")
            return True

        # Create temp directory
        temp_path.mkdir(exist_ok=True)
        env = os.environ.copy()

        # Disable HF_TRANSFER to avoid download errors
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

        # For HuggingFace models - download the entire folder
        if model_info["source"] == "huggingface":
            # Properly download the entire repository with all files
            # Don't use any include/exclude patterns - we want everything
            cmd = [
                "huggingface-cli", "download",
                model_info["repo"],
                "--local-dir", str(temp_path),
                "--local-dir-use-symlinks", "False",
                "--repo-type", "model"
            ]

            print_status(f"Downloading complete model repository: {sanitized_name}...", "info")
            print_status(f"Running command: {' '.join(cmd)}", "info")

            try:
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)

                # Check if the download seems successful by checking directory content
                if not list(temp_path.glob('**/*')):
                    print_status(f"Download completed but no files found in {temp_path}", "error")
                    print_status("Debug information:", "info")
                    subprocess.run(["ls", "-la", str(temp_path)], check=False)
                    return False

                # Print directory structure for debugging
                print_status("Downloaded model structure:", "info")
                print_model_structure(temp_path)

                # Move to final location
                if model_path.exists():
                    print_status(f"Model path {model_path} already exists, won't overwrite", "warning")
                    # Clean up the temp folder since we won't use it
                    shutil.rmtree(temp_path)
                    return True

                print_status(f"Moving from {temp_path} to {model_path}", "info")
                temp_path.rename(model_path)
                print_status(f"Successfully downloaded {sanitized_name}", "success")
                return True

            except subprocess.CalledProcessError as e:
                print_status(f"Failed to download model: {e.stderr}", "error")
                if temp_path.exists():
                    shutil.rmtree(temp_path)
                return False

        # Add logic for GGUF URL download
        elif model_info["source"] == "gguf_url":
            import requests
            import math

            url = model_info["repo"] # The URL is stored in the repo field
            filename = url.split('/')[-1] # Extract filename from URL
            model_file_path = temp_path / filename

            print_status(f"Downloading GGUF file from URL: {url}", "info")

            try:
                response = requests.get(url, stream=True, timeout=600) # 10 min timeout
                response.raise_for_status() # Raise an exception for bad status codes

                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024 # 1MB chunks
                downloaded_size = 0

                with open(model_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                # Simple progress update to console
                                sys.stdout.write(f"\rDownloading {filename}: {downloaded_size / block_size:.1f}/{total_size / block_size:.1f} MB ({progress:.1f}%)")
                                sys.stdout.flush()

                sys.stdout.write("\n") # New line after progress

                if downloaded_size == 0 or (total_size != 0 and downloaded_size != total_size):
                    print_status(f"Download incomplete. Expected {total_size} bytes, got {downloaded_size}", "error")
                    if model_file_path.exists(): model_file_path.unlink()
                    if temp_path.exists(): shutil.rmtree(temp_path)
                    return False

                # Move to final location
                if model_path.exists():
                    print_status(f"Model path {model_path} already exists, won't overwrite", "warning")
                    shutil.rmtree(temp_path)
                    return True

                print_status(f"Moving from {temp_path} to {model_path}", "info")
                # Ensure the final directory exists before renaming the file into it
                model_path.mkdir(exist_ok=True)
                model_file_path.rename(model_path / filename)
                # Remove the now-empty temp directory
                if temp_path.exists():
                     shutil.rmtree(temp_path)

                print_status(f"Successfully downloaded {sanitized_name}", "success")
                return True

            except requests.exceptions.RequestException as e:
                print_status(f"Failed to download GGUF file: {str(e)}", "error")
                if model_file_path.exists(): model_file_path.unlink()
                if temp_path.exists(): shutil.rmtree(temp_path)
                return False
            except Exception as e:
                print_status(f"An unexpected error occurred during GGUF download: {str(e)}", "error")
                if model_file_path.exists(): model_file_path.unlink()
                if temp_path.exists(): shutil.rmtree(temp_path)
                return False

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
                print_status(f"Model path {model_path} already exists, won't overwrite", "warning")
                # Clean up the temp folder since we won't use it
                shutil.rmtree(temp_path)
                return True

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

def model_is_large_enough(model_path: Path, min_size_gb: float = 10.0) -> bool:
    """Check if a model directory is large enough to be considered complete"""
    if not model_path.exists():
        return False

    # Check the size of the directory
    try:
        total_size = 0
        for path in model_path.glob('**/*'):
            if path.is_file():
                total_size += path.stat().st_size

        # Convert to GB
        size_gb = total_size / (1024**3)

        # If the model is larger than the minimum size, consider it complete
        if size_gb >= min_size_gb:
            print_status(f"Model {model_path.name} is {size_gb:.1f}GB - considered complete based on size", "info")
            return True

        print_status(f"Model {model_path.name} is only {size_gb:.1f}GB - may be incomplete", "warning")
        return False
    except Exception as e:
        print_status(f"Error checking model size: {str(e)}", "warning")
        return False

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

        # Check which models need to be downloaded - SIMPLIFIED
        for model_name, model_info in enabled_models.items():
            # Sanitize model name
            sanitized_name = model_name.strip('"\' \t')
            model_path = models_dir / sanitized_name

            # Simply check if the model folder exists - that's enough!
            if model_path.exists():
                print_status(f"Model {sanitized_name} already exists, skipping...", "info")
                continue
            else:
                print_status(f"Model {sanitized_name} is missing, will download", "warning")
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