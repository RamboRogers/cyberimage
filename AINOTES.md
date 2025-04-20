# Project Notes

## Current State
- **Removed IP-based rate limiting functionality** entirely:
    - Removed `@rate_limit` decorator from `app/api/routes.py`.
    - Deleted `app/utils/rate_limit.py`.
    - Removed related config from `instance/config.py` and `app/__init__.py`.
- Optimized image generation pipeline by removing forced model unloading after each job in `generator.py`.
- This allows model caching between jobs using the same model, significantly reducing processing time and estimated wait times.
- Investigated potential causes for long user wait times (identified aggressive memory cleanup as primary cause).

## Functions & APIs
- Refer to `API.md` for API definitions.
- `app/__init__.py`: Creates Flask app, loads config (from mapping and `instance/config.py`). *Rate limit config removed.*
- `instance/config.py`: Instance-specific configuration overrides. *Rate limit config removed.*
- `app/models/generator.GenerationPipeline`: Manages job queue and image generation.
  - `process_job`: Processes a single job (modified to keep model loaded).
  - `_force_memory_cleanup`: (Now less frequently used) Unloads models and clears GPU memory.
- `app/models.manager.ModelManager`: Handles loading, unloading, and generation with specific models.
- `app/utils.queue.QueueManager`: Manages job persistence and status in the database.
- `app/utils.image.ImageManager`: Handles image saving.
- `app/utils/config.py`: Contains utility functions.
- `app/static/js/main.js`: Handles frontend logic, including status polling and wait time estimation (`pollGenerationStatus`).

## Design Alignment (DESIGN.md)
- Current changes align with minimizing complexity and improving performance/user experience.

## Known Issues / TODOs
- Monitor performance after the optimization.
- Ensure memory management remains stable without forced cleanup after *every* job (Watchdog should help).