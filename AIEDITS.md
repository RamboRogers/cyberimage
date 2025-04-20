# AI Edits Log

## Edit 2024-07-27_02

- **File:** `app/utils/config.py`
- **Change:** Added `ENABLE_RATE_LIMIT = False` to disable the IP-based hourly rate limit.
- **Reason:** To remove the rate limiting restriction on the `/api/generate` endpoint as requested by the user.
- **Aligned with AINOTES:** Yes, updated notes reflect this configuration change.

## Edit 2024-07-27_01

- **File:** `app/models/generator.py`
- **Change:** Removed the call to `_force_memory_cleanup()` within the `finally` block of the `_generation_lock` in the `process_job` method.
- **Reason:** To improve performance by enabling model caching between jobs. The previous implementation unloaded the model after every job, causing significant delays due to model reloading, especially for consecutive jobs using the same model. This change allows the `ModelManager` to keep the last used model loaded, reducing job processing time and consequently the estimated wait times shown to the user.
- **Aligned with AINOTES:** Yes, updated notes reflect this performance optimization.

## Edit 2024-07-26_01 (Example Previous Edit)
... (other previous edits) ...