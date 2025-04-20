# Project Notes

## Current State
- **Disabled IP-based rate limiting** by setting `ENABLE_RATE_LIMIT = False` in `app/utils/config.py`. Users should no longer see 'Please try again in X seconds' errors due to hourly limits.
- Optimized image generation pipeline by removing forced model unloading after each job in `generator.py`.
- This allows model caching between jobs using the same model, significantly reducing processing time and estimated wait times.
- Investigated potential causes for long user wait times (identified aggressive memory cleanup as primary cause).

## Functions & APIs
- Refer to `API.md` for API definitions.
- `app.models.generator.GenerationPipeline`: Manages job queue and image generation.
  - `process_job`: Processes a single job (modified to keep model loaded).
  - `_force_memory_cleanup`: (Now less frequently used) Unloads models and clears GPU memory.
- `app.models.manager.ModelManager`: Handles loading, unloading, and generation with specific models.
- `app.utils.queue.QueueManager`: Manages job persistence and status in the database.
- `app.utils.image.ImageManager`: Handles image saving.
- `app.utils.rate_limit.py`: Defines `@rate_limit` decorator (now disabled via config).
- `app.utils.config.py`: Contains application configuration (added `ENABLE_RATE_LIMIT = False`).
- `app.static.js.main.js`: Handles frontend logic, including status polling and wait time estimation (`pollGenerationStatus`).

## Design Alignment (DESIGN.md)
- Current changes align with minimizing complexity and improving performance/user experience.

## Known Issues / TODOs
- Monitor performance after the optimization.
- Ensure memory management remains stable without forced cleanup after *every* job (Watchdog should help).