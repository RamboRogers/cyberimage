"""
Utility to clean up model directories with issues
"""
import os
from pathlib import Path
import shutil
import sys

def print_status(message, status="info"):
    """Print a status message"""
    status_icons = {
        "info": "ℹ️",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
    }
    icon = status_icons.get(status, "ℹ️")
    print(f"\n{icon} {message}")
    sys.stdout.flush()

def cleanup_models_directory(models_dir):
    """Clean up the models directory"""
    print_status(f"Cleaning up models directory: {models_dir}")

    # Ensure the directory exists
    if not models_dir.exists():
        print_status("Models directory does not exist", "error")
        return False

    # Find all problematic directories
    problem_dirs = []

    for path in models_dir.iterdir():
        if not path.is_dir():
            continue

        # Check if the directory name has quotes or ends with _temp
        if '"' in path.name or "'" in path.name or path.name.endswith("_temp"):
            problem_dirs.append(path)

    if not problem_dirs:
        print_status("No problematic directories found", "success")
        return True

    # Report the directories found
    print_status(f"Found {len(problem_dirs)} problematic directories:", "warning")
    for path in problem_dirs:
        print(f"  - {path.name}")

    # Ask for confirmation
    confirm = input("\nDelete these directories? (y/n): ").strip().lower()
    if confirm != 'y':
        print_status("Cleanup aborted", "info")
        return False

    # Delete the directories
    deleted = 0
    for path in problem_dirs:
        try:
            print_status(f"Removing {path.name}...", "info")
            shutil.rmtree(path)
            deleted += 1
        except Exception as e:
            print_status(f"Failed to remove {path.name}: {str(e)}", "error")

    print_status(f"Removed {deleted} of {len(problem_dirs)} directories", "success")
    return True

def main():
    """Main entry point"""
    # Get models directory from environment or use default
    models_dir = Path(os.getenv("MODEL_FOLDER", "./models"))

    print_status(f"Models Cleanup Utility\n\nModels directory: {models_dir}")
    cleanup_models_directory(models_dir)

if __name__ == "__main__":
    main()