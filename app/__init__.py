"""
CyberImage - Epic AI Image Generation Service
"""
import os
from flask import Flask, g, current_app, request
from flask_cors import CORS
from pathlib import Path
from app.utils.logging_config import setup_logging
from app.models.generator import GenerationPipeline
from contextlib import suppress
import sys
import signal

def create_app(test_config=None):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app)

    # Get directory configurations from environment or use defaults
    images_dir = os.getenv("IMAGES_FOLDER", "./images")
    models_dir = os.getenv("MODEL_FOLDER", "./models")

    # Ensure required directories exist
    images_path = Path(images_dir)
    images_path.mkdir(exist_ok=True)
    (images_path / "db").mkdir(exist_ok=True)  # Create db directory under images

    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)

    # Default configuration
    app.config.from_mapping(
        DATABASE=os.path.join(os.path.abspath(images_dir), "db", "cyberimage.sqlite"),
        IMAGES_PATH=os.path.abspath(images_dir),
        MODELS_PATH=os.path.abspath(models_dir),
        MAX_QUEUE_SIZE=10,
        MODEL_CACHE_SIZE=1,  # Only keep one model in memory at a time (memory constrained)
        ENABLE_RATE_LIMIT=True,
        RATE_LIMIT_HOURLY=10,
        ENABLE_XFORMERS=True,  # Enable memory efficient attention
        USE_HALF_PRECISION=True,  # Use FP16 for models when possible
        CLEANUP_INTERVAL=3600,  # Cleanup stalled jobs every hour
        MAX_PROMPT_LENGTH=500,
        DEFAULT_STEPS=30,
        DEFAULT_GUIDANCE_SCALE=7.5,
    )

    # Force single process mode in production
    if not app.debug:
        os.environ["GUNICORN_CMD_ARGS"] = "--workers=1"

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # Load the test config if passed in
        app.config.update(test_config)

    # Setup logging
    setup_logging(app)

    # Initialize database
    from app.utils import db
    db.init_app(app)

    # Download models if needed
    with app.app_context():
        from app.utils.download_models import download_all_models
        print("\n📥 Checking/Downloading required models...")
        sys.stdout.flush()
        download_all_models()

    def get_or_create_generator():
        """Get existing generator or create a new one if needed"""
        if not hasattr(current_app, '_generator'):
            current_app._generator = GenerationPipeline()
        return current_app._generator

    @app.before_request
    def before_request():
        """Ensure generator is available before API requests"""
        if request.endpoint and 'api.' in request.endpoint:
            g.generator = get_or_create_generator()

    # Replace the existing cleanup with a more targeted version
    def cleanup_handler(signum=None, frame=None):
        """Handle cleanup only on actual process termination"""
        try:
            if hasattr(current_app, '_generator'):
                generator = current_app._generator
                if generator and generator.is_running:
                    print("\n🛑 Performing final cleanup on process termination...")
                    sys.stdout.flush()
                    generator.stop()
                    print("✅ Cleanup completed")
                    sys.stdout.flush()
        except Exception as e:
            print(f"\n⚠️  Final cleanup warning: {str(e)}")
            sys.stdout.flush()

    # Register the cleanup handler for process termination
    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)

    # Register blueprints
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix="/api")

    # Register main blueprint for frontend routes
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    # Health check endpoint
    @app.route("/health")
    def health_check():
        generator = get_or_create_generator()
        try:
            from app.utils.queue import QueueManager
            queue_status = QueueManager.get_queue_status()
            return {
                "status": "healthy",
                "version": "1.0.0",
                "device": generator.model_manager._device if generator.model_manager else "not_initialized",
                "queue": {
                    "in_memory_size": generator.generation_queue.qsize() if generator.generation_queue else 0,
                    "pending": queue_status["pending"],
                    "processing": queue_status["processing"],
                    "completed": queue_status["completed"],
                    "failed": queue_status["failed"],
                    "total": queue_status["total"]
                },
                "is_main_process": generator.is_main_process,
                "is_running": generator.is_running
            }
        except Exception as e:
            app.logger.error(f"Error in health check: {str(e)}")
            return {
                "status": "degraded",
                "version": "1.0.0",
                "error": str(e),
                "queue": {
                    "in_memory_size": generator.generation_queue.qsize() if generator.generation_queue else 0
                },
                "is_main_process": generator.is_main_process,
                "is_running": generator.is_running
            }

    app.logger.info("Application initialized successfully")
    return app