# CyberImage Project Notes

## Project State
- Integrating distinct Text-to-Video and Image-to-Video generation capabilities.
- Added `/generate_t2v` endpoint.
- Clarified `/generate_video` endpoint for Image-to-Video.
- **Video Generation Pipeline Simplification:** Refactored video generation (T2V and I2V) to be API-only.

## Core Components Tracking
### Storage Structure
- /images - Local image storage
- /db - SQLite database storage
- /videos - Local video storage (Implicitly handled by ImageManager for now)

### API Endpoints (REST)
- POST /api/generate - Submit Text-to-Image generation request
- POST /api/generate_t2v - Submit Text-to-Video generation request (New)
- POST /api/generate_video - Submit Image-to-Video generation request (Previously just "Video")
- GET /api/status/{job_id} - Check job status (handles image, t2v, i2v jobs)
- GET /api/image/{image_id} - Retrieve generated image
- GET /api/video/{video_id} - Retrieve generated video (New/Refined)
- GET /api/models - List available models (incl. video types)
- GET /api/queue - View current queue status

### Database Schema
#### Jobs Table
```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    status TEXT,  -- PENDING, PROCESSING, COMPLETED, FAILED
    model TEXT,
    prompt TEXT,  -- Main prompt (text for T2I/T2V, video prompt for I2V)
    created_at TIMESTAMP,
    completed_at TIMESTAMP,
    media_path TEXT, -- Renamed from image_path to handle both
    settings JSON -- Includes type ('image', 't2v', 'i2v'), source_image_id for i2v, etc.
);
```
#### Images/Media Table
```sql
CREATE TABLE images ( -- May need renaming to 'media' eventually
    id TEXT PRIMARY KEY,
    job_id TEXT,
    file_path TEXT,
    created_at TIMESTAMP,
    metadata JSON, -- includes original settings, prompt, model, potentially media type ('image', 'video')
    FOREIGN KEY(job_id) REFERENCES jobs(id)
);
```
*(Note: DB Schema updates are proposed, need implementation)*

### Models Integration
Available Models:
1. FLUX.1-dev (black-forest-labs/FLUX.1-dev) - Image
2. SD 3.5 Large (stabilityai/stable-diffusion-3.5-large) - Image
3. Flux.1dev Abliterated (aoxo/flux.1dev-abliteratedv2) - Image
4. wan-t2v (e.g., `wan-t2v` from .env) - Text-to-Video (T2V)
5. Wan-AI/Wan2.1-I2V-14B-480P (e.g., `MODEL_I2V` from .env) - Image-to-Video (I2V)

*(Note: Actual IDs and how they are parsed from .env need confirmation in `app/models/__init__.py`)*

#### Proposed .env Format Update (Model Configuration)
(Existing proposal remains relevant, may need a 'type' field if not inferred from repo)

### Core Functions
```python
# Existing
def generate_image(prompt: str, model_id: str, settings: dict) -> str:
    \"\"\"Generate image and return job_id\"\"\"

def get_job_status(job_id: str) -> dict:
    \"\"\"Get status of generation job (image or video)\"\"\"

def get_image(image_id: str) -> str:
    \"\"\"Get path to generated image\"\"\"

def process_queue() -> None:
    \"\"\"Process pending generation jobs (image or video)\"\"\"

# New/Updated
def generate_t2v(prompt: str, model_id: str, settings: dict) -> str:
    \"\"\"Generate text-to-video and return job_id\"\"\"

def generate_i2v(source_image_id: str, video_prompt: str, model_id: str, settings: dict) -> str:
    \"\"\"Generate image-to-video and return job_id\"\"\"

def get_video(video_id: str) -> str:
    \"\"\"Get path to generated video\"\"\"

# Backend API Handlers (app/api/routes.py)
def generate_image(): # Handles POST /api/generate
    pass
def generate_t2v(): # Handles POST /api/generate_t2v (New)
    pass
def generate_video(): # Handles POST /api/generate_video (Now specifically I2V)
    pass
def get_job_status(job_id): # Handles GET /api/status/{job_id}
    pass
def get_image(image_id): # Handles GET /api/image/{image_id}
    pass
def get_video(video_id): # Handles GET /api/video/{video_id} (New/Refined)
    pass
def get_models(): # Handles GET /api/models
    pass
def get_queue(): # Handles GET /api/queue
    pass
# ... other handlers ...
```

## Frontend Enhancements (main.js)
- **Main Form (T2I/T2V Handling):**
  - Modified `initializeModels` to populate the main model dropdown (`#model`) with both T2I ('image') and T2V ('t2v') models.
  - Modified `handleModelChange` to update the generate button text (⚡ Generate Image / 🎬 Generate Video) based on the selected model's type.
  - Updated the form submission logic in `initializeGenerationForm`:
    - Detects selected model type (T2I or T2V).
    - Sets the target API endpoint (`/api/generate` or `/api/generate_t2v`).
    - Constructs the appropriate request payload for the endpoint.
    - Uses a `feedbackType` variable ('Image' or 'Video') for UI messages.
  - Updated `pollGenerationStatus` to accept and use `feedbackType` for more accurate messaging.
- **localStorage Persistence (Existing):**
  - Keeps track of last selected model, steps, guidance, keep prompt preference, and kept prompt text.
- **Dynamic Step/Guidance Config (Existing):**
  - Adjusts slider ranges/values based on `step_config` from model data.
- **Video Generation Modal (I2V):**
  - Logic for the separate Image-to-Video modal remains unchanged.

## Technical Decisions
- Python 3.12 + Flask Backend
- SQLite for data persistence and queue management
- Local filesystem for image/video storage
- REST API only, no WebSocket
- No authentication required
- Models loaded on-demand to manage memory

## Progress Log
- Added `/generate_t2v` endpoint for Text-to-Video.
- Updated `/generate_video` endpoint to specifically handle Image-to-Video.
- Updated `API.md` documentation.
- Updated `NOTES.md` state.
- Added `MODEL_I2V` to `.env` (manual step for user).
- Fixed I2V model detection in `main.js` to recognize models with "i2v" in their ID even if type property wasn't explicitly set to "i2v".
- **Updated main generation form (`main.js`) to support both T2I and T2V models.**

## Current Goal
- Implement the backend logic in `GenerationPipeline` to handle the different job types (`image`, `t2v`, `i2v`) based on `settings['type']`.
- Ensure `AVAILABLE_MODELS` in `app/models/__init__.py` correctly loads and identifies T2V and I2V models based on `.env` configuration.
- Update frontend (`main.js`, `index.html`) to allow selection between Text-to-Image, Text-to-Video, and initiating Image-to-Video generation.
- **Verify frontend (`main.js`, `index.html`) correctly allows selection between Text-to-Image and Text-to-Video from the main form.**

## Design Decisions (Video Feature Refined)
- **Models:**
    - Text-to-Video: `wan-t2v` (via `MODEL_T2V` env var). Triggered from main form (new option).
    - Image-to-Video: `Wan-AI/Wan2.1-I2V-14B-480P` (via `MODEL_I2V` env var). Triggered from existing image (e.g., gallery 🎥 button).
- **API:**
    - `/generate`: Text-to-Image only.
    - `/generate_t2v`: Text-to-Video only. Needs `model_id` (T2V type), `prompt`, `settings`.
    - `/generate_video`: Image-to-Video only. Needs `source_image_id`, `video_model_id` (I2V type), `video_prompt`, `settings`.
- **Backend:**
    - `GenerationPipeline.process_job` needs branching logic based on `job['settings']['type']`.
    - `QueueManager` needs to store job type (via `settings`).
    - `ImageManager` might need adaptation to handle video paths/metadata, or a separate `VideoManager` created. (Current approach: use `ImageManager` with video paths/metadata).
- **Database:**
    - `jobs` table needs `settings['type']` reliably set.
    - `jobs.media_path` stores output path (either .png or .mp4).
    - `images` table stores metadata; may need a `media_type` column eventually.
- **Frontend:**
    - Main form: Add option/dropdown to select output type (Image or Video (T2V)). Model list should filter accordingly.
    - Gallery: Keep 🎥 button on images for I2V. Modal needs to prompt for `video_prompt` and select an I2V model.

## Model Manager (`manager.py`) Updates (Revised)
- Need methods like `generate_text_to_video(...)` and `generate_image_to_video(...)`.
- Need to handle loading/unloading of T2V and I2V models distinctly.
- Imports for specific pipelines (e.g., `DiffusionPipeline` for T2V, `WanImageToVideoPipeline` for I2V).
- Corrected `get_model` to load based on `i2v` or `t2v` type from config.
- Renamed `generate_video` -> `generate_image_to_video`.
- Added placeholder `generate_text_to_video`.

## Function List (Updated)
(See Core Functions section above - note ModelManager changes)

## Model Issues and Fixes
- **I2V Model Detection:**
  - Issue: Models with IDs containing "i2v" (e.g., "wan-i2v-14b") weren't being recognized when the model's type property wasn't explicitly set to "i2v".
  - Fix: Updated `openVideoGenModal()` in `main.js` to detect I2V models using two criteria:
    1. Model's type property is set to 'i2v' OR
    2. Model ID contains 'i2v' (case insensitive)
  - This accommodates both properly typed models from the server and models that are only identifiable by their ID.
  - Added fallback description text for models missing description field.

- **LTX-Video Output Corruption:**
  - Issue: Video output from LTX-Video model (T2V) was reported as corrupt (e.g., "green lines"), then distorted but recognizable figures.
  - Investigation 1: Logs showed `DEBUG: Before export_to_video - frames type is not ndarray: <class 'list'>`.
  - Root Cause 1: The `generate_text_to_video` function returned a list, but `export_to_video` expects NumPy array.
  - Fix 1: Added NumPy conversion in `generator.py`. (Later removed)
  - Investigation 2: Output still corrupt after Fix 1.
  - Potential Root Cause 2a: Mismatch between dtypes used in `get_model` vs reference.
  - Fix 2a: Modified `get_model` in `manager.py` to strictly use `torch.bfloat16`.
  - Potential Root Cause 2b: Frames returned might be tensors, causing incorrect conversion.
  - Fix 2b: Added debug logging in `generate_text_to_video` (`manager.py`) to inspect frame format.
  - Investigation 3: Generation failed with `NameError: name 'Image' is not defined` in debug log.
  - Root Cause 3: Missing `PIL.Image` import in `manager.py`.
  - Fix 3: Added `from PIL import Image` to `manager.py`.
  - Investigation 4: Removed explicit NumPy conversion from `generator.py` to align flow with reference. Video now visible but distorted.
  - Potential Root Cause 4: Missing `negative_prompt`. Application default was `None` if empty, reference uses specific quality prompt.
  - Fix 4: Modified `generate_text_to_video` in `manager.py` to use reference negative prompt (`"worst quality..."`) as default if none provided by user.
  - Status: LTX-Video aligned with reference negative prompt. Explicit conversion removed. Debug logging remains in `manager.py` `generate_text_to_video`.

- **I2V Model Loading Failure:**
  - Issue: I2V job failed with `OSError` and `LocalEntryNotFoundError` when loading `image_encoder`.
  - Root Cause: Required model component files were missing locally, and loading code used `local_files_only=True`, preventing download.
  - Fix: Changed `local_files_only=True` to `local_files_only=False` for the I2V component loading calls (`CLIPVisionModel`, `AutoencoderKLWan`, `WanImageToVideoPipeline`) in `app/models/manager.py` (`get_model` function).
  - Status: Implemented. Requires testing I2V again (will download missing files on first run).

- **I2V Model Loading Failure (Update):**
  - Issue: I2V job still failed with `OSError: ... does not appear to have a file named pytorch_model.bin...` even after allowing downloads.
  - Root Cause 1: The configured repository ID in `.env` (`Wan-AI/Wan2.1-I2V-14B-480P`) was likely incorrect. The reference uses `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`, which has the expected file structure.
  - Root Cause 2: VAE was loaded using `model_path` instead of the repository ID, inconsistent with other components.
  - Fix 1 (Manual): User needs to update `.env` file to set the `wan-i2v-14b` model repo to `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`.
  - Fix 2 (Code): Changed `AutoencoderKLWan.from_pretrained` in `manager.py` to load from `model_config['repo']` instead of `model_path`.
  - Status: Code fix applied. Manual `.env` update required by user. Requires testing I2V again.

- **I2V Job Routing Failure:**
  - Issue: I2V job was processed by the Image Generation logic (`generate_image`) instead of I2V logic, causing `TypeError: WanImageToVideoPipeline.__call__() missing 1 required positional argument: 'image'`.
  - Root Cause: The `/generate_video` API route in `app/api/routes.py` was setting the job type as `"video"` instead of the expected `"i2v"`.
  - Fix: Updated the `/generate_video` route to set `settings['type'] = 'i2v'`.
  - Status: Implemented.

- **I2V Out-of-Memory (OOM) Error:**
  - Issue: I2V job failed with `CUDA out of memory` error during generation.
  - Root Cause: The generation resolution calculated based on the source image (e.g., 960x960) was too large for the `wan-i2v-14b` model within the available VRAM (24GB), even with CPU offloading.
  - Fix: Modified the call to `generate_image_to_video` in `app/models/generator.py` to explicitly pass `max_video_area=480*832`. This forces the image pre-processing step (`_aspect_ratio_resize`) to use a smaller target resolution, similar to the reference code, reducing memory usage.
  - Status: Implemented. Requires testing I2V again.

- **Integrate LTX for I2V (Experimental GGUF Approach):**
  - Goal: Allow using the LTX GGUF model for Image-to-Video.
  - Method: Attempt to load `LTXImageToVideoPipeline` by injecting the GGUF transformer (deviation from LTX I2V reference code).
  - Fix 1: Modified `get_model` in `manager.py` to check if `model_type` is `i2v` and `source` is `gguf_url`. If so, load GGUF transformer and attempt to pass it to `LTXImageToVideoPipeline.from_pretrained`. Logs warning about experimental nature.
  - Fix 2: Modified `generate_image_to_video` in `manager.py` to check `isinstance(pipe, LTXImageToVideoPipeline)`. If true, skip `_aspect_ratio_resize`, use LTX reference parameters (width, height, frames, steps, negative prompt default), and call the pipeline. Otherwise, use existing Wan I2V logic.
  - Issue: Loading failed with `NameError: name 'LTXImageToVideoPipeline' is not defined`.
  - Root Cause: Missing import for `LTXImageToVideoPipeline` in `manager.py`.
  - Fix 3: Added `LTXImageToVideoPipeline` to the `diffusers` import statement in `manager.py`.
  - Status: Implemented. Requires testing LTX GGUF model with an I2V job.

- **Media Deletion Failure on Index Page:**
  - Issue: Delete button worked on gallery page but failed with "NOT FOUND" on index page (recent generations).
  - Root Cause: Delete handler in `main.js` (used by index page) constructed the API URL dynamically (`/api/${mediaType}/${mediaId}`). This failed for videos as `/api/video/...` doesn't exist for DELETE. The handler in `gallery.js` always used `/api/image/...`.
  - Fix: Modified the delete handler in `main.js` to always use the `/api/image/${mediaId}` endpoint for deletion, regardless of `mediaType`.
  - Status: Implemented. Requires testing delete on index page.

## UI Updates (Revised)
- **`index.html` / `main.js`:**
  - **Main form now supports T2I and T2V model selection.**
  - Model dropdown filters to show only T2I/T2V models.
  - Generate button text updates dynamically.
  - Form submits to correct API endpoint based on model type.
- **`gallery.html` / `main.js`:**
  - Ensure 🎥 button triggers I2V flow (modal for video prompt, select I2V model).
- Display videos correctly in gallery/modals (`<video>` tag).

## Database Considerations (Revised)
- `jobs` table needs `settings['type']` reliably set.
- Consider adding `media_type` ('image'/'video') to `images` table long-term.

## Video Generation Issues
- **LTX-Video Output Corruption:**
  - Issue: Video output from LTX-Video model (T2V) was reported as corrupt (e.g., "green lines"), then distorted but recognizable figures.
  - Investigation 1: Logs showed `DEBUG: Before export_to_video - frames type is not ndarray: <class 'list'>`.
  - Root Cause 1: The `generate_text_to_video` function returned a list, but `export_to_video` expects NumPy array.
  - Fix 1: Added NumPy conversion in `generator.py`. (Later removed)
  - Investigation 2: Output still corrupt after Fix 1.
  - Potential Root Cause 2a: Mismatch between dtypes used in `get_model` vs reference.
  - Fix 2a: Modified `get_model` in `manager.py` to strictly use `torch.bfloat16`.
  - Potential Root Cause 2b: Frames returned might be tensors, causing incorrect conversion.
  - Fix 2b: Added debug logging in `generate_text_to_video` (`manager.py`) to inspect frame format.
  - Investigation 3: Generation failed with `NameError: name 'Image' is not defined` in debug log.
  - Root Cause 3: Missing `PIL.Image` import in `manager.py`.
  - Fix 3: Added `from PIL import Image` to `manager.py`.
  - Investigation 4: Removed explicit NumPy conversion from `generator.py` to align flow with reference. Video now visible but distorted.
  - Potential Root Cause 4: Missing `negative_prompt`. Application default was `None` if empty, reference uses specific quality prompt.
  - Fix 4: Modified `generate_text_to_video` in `manager.py` to use reference negative prompt (`"worst quality..."`) as default if none provided by user.
  - Status: Fix 2a, Fix 2b (logging), Fix 3, Fix 4 implemented. Explicit conversion removed. Requires testing.