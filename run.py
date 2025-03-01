"""
CyberImage Application Entry Point

Sets up proper system configuration for PyTorch memory management
and ensures clean shutdown of GPU resources.
"""
import os
import sys
import signal
import atexit
import gc
import threading
import time
from app import create_app

# --- PyTorch Memory Management Configuration ---
# Allow memory to be allocated more efficiently (prevents fragmentation)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
# Restrict to only use one GPU (device 0) if multiple are available
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Add optimization for memory allocation/deallocation
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Enable tensor debugging when needed
# os.environ['PYTORCH_DEBUG'] = '1'

# --- Create the Flask application ---
app = create_app()

# Ensure the generator is properly initialized
with app.app_context():
    from app.models.generator import GenerationPipeline

    # Store reference to the generator for cleanup
    if not hasattr(app, '_generator'):
        print("\n🔄 Initializing GenerationPipeline during application startup")
        app._generator = GenerationPipeline()
        print("✅ GenerationPipeline initialized")
        sys.stdout.flush()

# --- Register cleanup handlers ---
def cleanup_handler(signum=None, frame=None):
    """
    Comprehensive cleanup handler for application shutdown

    This ensures all GPU resources are properly released on exit
    """
    print("\n🧹 Application shutdown - performing comprehensive cleanup...")
    sys.stdout.flush()

    try:
        # Set a flag to prevent new requests
        app.shutting_down = True

        # Access model manager if available in app context
        with app.app_context():
            # Try to access the generator through the app
            if hasattr(app, '_generator'):
                print("✅ Stopping generator pipeline...")
                app._generator.stop()
                # Give it some time to complete cleanup
                time.sleep(1)
                sys.stdout.flush()

            # Force CUDA cleanup if available
            try:
                import torch
                if torch.cuda.is_available():
                    print("✅ Forcing CUDA memory cleanup...")
                    sys.stdout.flush()
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()

                    # Second pass cleanup
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Report final memory state
                    memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                    print(f"✅ Final GPU memory: {memory_allocated:.2f}GB")
                    sys.stdout.flush()
            except Exception as cuda_error:
                print(f"⚠️ CUDA cleanup warning: {str(cuda_error)}")
                sys.stdout.flush()

    except Exception as e:
        print(f"❌ Error during shutdown cleanup: {str(e)}")
        sys.stdout.flush()

    print("👋 Application shutdown complete!")
    sys.stdout.flush()

    # Exit if called as signal handler
    if signum is not None:
        sys.exit(0)

# Register the cleanup handler for normal exit
atexit.register(cleanup_handler)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, cleanup_handler)
signal.signal(signal.SIGTERM, cleanup_handler)

# If running on Linux, also handle SIGQUIT
if hasattr(signal, 'SIGQUIT'):
    signal.signal(signal.SIGQUIT, cleanup_handler)

if __name__ == "__main__":
    print("\n🚀 Starting CyberImage with optimized GPU memory management")
    print("⚠️ PRODUCTION NOTE: Using a single worker is REQUIRED for this application")
    print("   Example: gunicorn -w 1 -b 0.0.0.0:5050 run:app")
    print("⚠️ Multiple workers will cause memory/generation conflicts!")
    sys.stdout.flush()

    # For development, use threaded=True with a single process
    # The threaded Flask server is fine since we have a dedicated generation thread
    app.run(debug=True, host="0.0.0.0", port=5050, use_reloader=False, threaded=True)