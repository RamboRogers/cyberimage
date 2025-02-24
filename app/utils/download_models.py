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
    "flux-2": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "fp8_weights": {
            "repo": "Comfy-Org/flux1-dev",
            "file": "flux1-dev-fp8.safetensors"
        },
        "description": "FLUX model optimized for memory efficiency (FP8)",
        "requires_auth": True,
        "source": "huggingface",
        "files": [
            "model_index.json",
            "ae.safetensors",
            "flux1-dev.safetensors"
        ],
        "local_path": "models/flux-2"
    },
    "flux-1": {
        "repo": "black-forest-labs/FLUX.1-dev",
        "description": "FLUX base model",
        "requires_auth": True,
        "source": "huggingface",
        "files": [
            "model_index.json",
            "ae.safetensors",
            "flux1-dev.safetensors"
        ]
    },
    "sd-3.5": {
        "repo": "stabilityai/stable-diffusion-3.5-large",
        "description": "Stable Diffusion 3.5",
        "requires_auth": True,
        "source": "huggingface",
        "files": [
            "model_index.json",
            "sd3.5_large.safetensors"
        ]
    },
    "flux-abliterated": {
        "repo": "aoxo/flux.1dev-abliteratedv2",
        "description": "FLUX Abliterated variant",
        "requires_auth": True,
        "source": "huggingface",
        "files": [
            "model_index.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
            "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors"
        ]
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
    """Download a specific model using either HuggingFace CLI or Civitai API"""
    try:
        print_status(f"Downloading {model_name}: {model_info['description']}", "pending")

        model_path = models_dir / model_name
        temp_path = models_dir / f"{model_name}_temp"

        # Clean up any leftover temp directory
        if temp_path.exists():
            shutil.rmtree(temp_path)

        # Special handling for flux-2 - copy from flux-1
        if model_name == "flux-2":
            flux1_path = models_dir / "flux-1"
            if not flux1_path.exists():
                print_status("flux-1 model not found - needed as base for flux-2", "error")
                return False

            # Make sure temp directory is completely clean
            if temp_path.exists():
                print_status("Cleaning up temporary directory...", "info")
                shutil.rmtree(temp_path)

            # Also clean up target if it exists
            if model_path.exists():
                print_status("Cleaning up existing model directory...", "info")
                shutil.rmtree(model_path)

            # Copy flux-1 structure to temp directory
            print_status("Copying base model structure from flux-1...", "info")
            shutil.copytree(flux1_path, temp_path, dirs_exist_ok=True)

            # Download FP8 weights
            fp8_info = model_info["fp8_weights"]
            print_status(f"Downloading FP8 weights from {fp8_info['repo']}...", "info")

            # Download FP8 weights using snapshot-download for direct file access
            cmd = ["huggingface-cli", "download", "--repo-type=model",
                  fp8_info["repo"],
                  "--local-dir", str(temp_path),
                  "--local-dir-use-symlinks", "False"]

            env = os.environ.copy()
            env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

            if result.returncode != 0:
                print_status(f"Failed to download FP8 weights: {result.stderr}", "error")
                shutil.rmtree(temp_path)
                return False

            # Verify the FP8 file exists
            fp8_path = temp_path / fp8_info["file"]
            if not fp8_path.exists():
                print_status(f"FP8 weights file not found at {fp8_path}", "error")
                shutil.rmtree(temp_path)
                return False

            # Replace the model weights with FP8 version
            target_path = temp_path / "flux1-dev.safetensors"
            try:
                shutil.move(str(fp8_path), str(target_path))
                print_status("Successfully moved FP8 weights to correct location", "success")
            except Exception as e:
                print_status(f"Failed to move FP8 weights: {str(e)}", "error")
                shutil.rmtree(temp_path)
                return False

            # Move to final location
            if model_path.exists():
                shutil.rmtree(model_path)
            temp_path.rename(model_path)
            print_status(f"Successfully downloaded {model_name}", "success")
            return True

        # Normal download path for other models
        temp_path.mkdir(exist_ok=True)
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        if model_info["source"] == "civitai":
            # Extract model ID from repo string (format: "civitai:123456")
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

        else:  # Hugging Face download
            # First download base model structure (excluding safetensors)
            cmd = ["huggingface-cli", "download",
                  model_info["repo"],
                  "--local-dir", str(temp_path),
                  "--include", "*.json",  # Get all config files
                  "--include", "*.txt",   # Get any text files
                  "--include", "*.bin"]   # Get any binary files

            print_status(f"Downloading {model_name} base structure...", "info")
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

            if result.returncode != 0:
                print_status(f"Failed to download base model: {result.stderr}", "error")
                shutil.rmtree(temp_path)
                return False

            # If this model has FP8 weights, download them
            if "fp8_weights" in model_info:
                fp8_info = model_info["fp8_weights"]
                print_status(f"Downloading FP8 weights from {fp8_info['repo']}...", "info")

                # Download FP8 weights using snapshot-download for direct file access
                cmd = ["huggingface-cli", "download", "--repo-type=model",
                      fp8_info["repo"],
                      "--local-dir", str(temp_path),
                      "--local-dir-use-symlinks", "False"]

                result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

                if result.returncode != 0:
                    print_status(f"Failed to download FP8 weights: {result.stderr}", "error")
                    shutil.rmtree(temp_path)
                    return False

                # Verify the FP8 file exists
                fp8_path = temp_path / fp8_info["file"]
                if not fp8_path.exists():
                    print_status(f"FP8 weights file not found at {fp8_path}", "error")
                    shutil.rmtree(temp_path)
                    return False

                # Move FP8 weights to correct location
                target_path = temp_path / "flux1-dev.safetensors"
                try:
                    shutil.move(str(fp8_path), str(target_path))
                    print_status("Successfully moved FP8 weights to correct location", "success")
                except Exception as e:
                    print_status(f"Failed to move FP8 weights: {str(e)}", "error")
                    shutil.rmtree(temp_path)
                    return False

            # Download any remaining files (like ae.safetensors)
            cmd = ["huggingface-cli", "download",
                  model_info["repo"],
                  "--local-dir", str(temp_path),
                  "--include", "ae.safetensors"]

            print_status("Downloading additional model files...", "info")
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

            # Move to final location
            if model_path.exists():
                shutil.rmtree(model_path)
            temp_path.rename(model_path)
            print_status(f"Successfully downloaded {model_name}", "success")
            return True

    except subprocess.CalledProcessError as e:
        print_status(f"Failed to download {model_name}: {e.stderr}", "error")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        return False
    except Exception as e:
        print_status(f"Failed to download {model_name}: {str(e)}", "error")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        return False

def check_model_files(model_path: Path, files: list) -> bool:
    """Check if all required files exist for a model"""
    return all((model_path / file).exists() for file in files)

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

            # Check if model directory exists and has all required files
            if model_path.exists() and check_model_files(model_path, model_info["files"]):
                print_status(f"Model {model_name} already exists, skipping...", "info")
                continue

            # If we get here, the model needs to be downloaded
            print_status(f"Model {model_name} is missing or incomplete", "warning")
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