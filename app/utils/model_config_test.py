"""
Test script for model configuration
"""
import os
import sys
from config import parse_model_config, get_downloadable_models, get_available_models

def main():
    print("\n===== Testing Model Configuration =====\n")

    # Test environment variables
    os.environ["MODEL_1"] = "test-model-1;repo/test;Test Model;huggingface;true"
    os.environ["MODEL_2"] = "test-model-2;org/repo;Another Model;huggingface;true"
    os.environ["DOWNLOAD_MODEL_2"] = "false"

    # Test parsing model configuration
    print("Parsing model configuration...")
    models = parse_model_config()

    print(f"\nFound {len(models)} models:")
    for name, config in models.items():
        print(f"- {name}: {config['description']} ({config['repo']})")
        print(f"  Source: {config['source']}, Download enabled: {config['download_enabled']}")
        print(f"  Type: {config['type']}")
        print(f"  Files to check: {len(config.get('files', []))}")

    # Test getting downloadable models
    print("\nGetting downloadable models...")
    download_models = get_downloadable_models()
    print(f"Found {len(download_models)} downloadable models:")
    for name in download_models:
        print(f"- {name}")

    # Test getting available models for UI
    print("\nGetting available models for UI...")
    available_models = get_available_models()
    print(f"Found {len(available_models)} available models:")
    for name, config in available_models.items():
        print(f"- {name} (id: {config['id']})")

    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()