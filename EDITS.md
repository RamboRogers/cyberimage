# CyberImage Code Change Log

## Change History

### [2025-05-17] Initial Setup
- Created NOTES.md for project tracking
- Created EDITS.md for change logging
- Updated project design with specific models and storage decisions

### [2025-05-17] Phase 1 Implementation
- Created project directory structure
- Set up requirements.txt with carefully selected dependencies
- Initialized Flask application with:
  - CORS support
  - Configuration management
  - Health check endpoint
- Created API blueprint with initial routes
- Defined available models configuration
- Created application entry point

### [2025-05-17] Phase 2 Implementation
- Created database initialization system with SQLite
- Implemented queue management system with:
  - Job creation and status tracking
  - Queue status monitoring
  - Stalled job cleanup
- Implemented model management system with:
  - Dynamic model loading/unloading
  - GPU support
  - Memory management
- Updated main application to use new components

### [2025-05-17] Phase 3 Implementation
- Created image management system with:
  - Organized storage structure (by date)
  - Database tracking
  - Metadata support
- Implemented comprehensive API endpoints:
  - POST /api/generate - Submit generation requests
  - GET /api/status/<job_id> - Check job status
  - GET /api/image/<image_id> - Retrieve generated images
  - GET /api/image/<image_id>/metadata - Get image metadata
  - GET /api/queue - View queue status
- Added input validation and error handling
- Integrated all components (Queue, Model, Image managers)

### [2025-05-17] Phase 4 Implementation
- Created generation pipeline:
  - Job processing system
  - Error handling and recovery
  - Automatic cleanup of stalled jobs
- Enhanced model manager:
  - Improved error handling
  - Memory optimization
  - Performance improvements
  - Detailed logging
- Implemented worker process:
  - Background job processing
  - Graceful shutdown handling
  - Resource cleanup
- Added generation optimizations:
  - Memory efficient attention
  - Inference mode
  - Default parameters

### [2025-05-17] Phase 5 Implementation
- Implemented comprehensive logging system:
  - Rotating file handlers
  - Separate logs for app, errors, and models
  - Structured log formats
  - Log level configuration
- Added rate limiting:
  - Per-IP request tracking
  - Configurable limits
  - Automatic cleanup
- Enhanced error handling:
  - Custom API error class
  - Consistent error responses
  - Detailed error logging
- Added performance optimizations:
  - Memory efficient attention (xformers)
  - Half-precision (FP16) support
  - Configurable cleanup intervals
  - Optimized model loading
- Improved configuration:
  - Added rate limit settings
  - Performance tuning options
  - Default generation parameters

### [2025-05-17] Frontend Enhancements (localStorage)
- Modified `app/static/js/main.js`:
  - Added logic to `initializeModels` to load `lastModelId` from `localStorage` and apply it to the model select.
  - Added logic to model select `change` event listener to save the selected `modelId` to `localStorage`.
  - Added logic to `initializeGenerationForm` to load `lastStepsValue` from `localStorage` and apply it to the steps slider and display.
  - Added event listener to steps slider (`input`) to save its value to `localStorage`.
  - Added logic to form `submit` event listener to save both current `modelId` and `stepsValue` to `localStorage`.

### [2025-05-17] Backend Model Step Configuration
- Updated `.env.example` and `.env`:
  - Modified format comment to include optional `<options_json>`.
  - Added `MODEL_4` entry for `sana-sprint` with `{\"fixed_steps\": 2}` configuration.
- Modified `app/utils/config.py` (`parse_model_config` function):
  - Updated logic to parse the optional 6th semicolon-separated part as a JSON string.
  - Added JSON parsing with error handling for the options string.
  - Stored the parsed dictionary (or empty dict if invalid/missing) into the `step_config` key for each model in `models_config`.
- Verified `app/api/routes.py` (`/api/models` endpoint): Confirmed it returns the `AVAILABLE_MODELS` dictionary directly, which now includes the `step_config` data. No changes needed.

### [2025-05-17] Frontend Step Configuration Handling
- Modified `app/static/js/main.js`:
  - In `initializeModels`:
    - Created `modelsDataStore` to hold the full configuration fetched for each model.
    - Populated `modelsDataStore` while creating model select options.
  - In the `#model` select `change` event listener:
    - Retrieved the full data object (`selectedModelData`) for the chosen model ID from `modelsDataStore`.
    - Accessed the `step_config` from `selectedModelData`.
    - Added logic to check `step_config` for `fixed_steps` or a `steps` range (`min`, `max`, `default`).
    - Updated the `#steps` slider's `min`, `max`, `value`, and `disabled` attributes based on the configuration.
    - Prioritized `localStorage` value for steps if it's within the allowed range (for non-fixed models).
    - Updated the `#steps-value` display.
    - Saved the potentially adjusted steps value back to `localStorage`.
    - Added fallback to reset slider to defaults if no model is selected or data is missing.

## Implementation Plan

### Phase 1: Basic Infrastructure
- [x] Create project directory structure
- [x] Set up requirements.txt
- [x] Initialize Flask application
- [x] Create necessary directories (db/, images/)

### Phase 2: Core Components
- [x] Database initialization script
- [x] Model manager implementation
- [x] Queue system setup
- [x] Basic Flask application with health check

### Phase 3: API Implementation
- [x] Generation endpoint
- [x] Status checking endpoint
- [x] Image retrieval endpoint
- [x] Queue status endpoint

### Phase 4: Model Integration
- [x] Model loading system
- [x] Generation pipeline
- [x] Image saving and management

### Phase 5: Testing & Optimization
- [x] Error handling
- [x] Logging implementation
- [x] Rate limiting
- [x] Model loading optimization

## Current Focus
Completed Phase 5: Testing & Optimization
All planned phases are now complete!

## [2025-05-17] - Add Video Generation Feature (Initial Phase)

- **API.md:** Added documentation for new `/api/generate_video` endpoint and corresponding `GET /get_video/<video_id>`.
- **NOTES.md:** Documented design, model details, API endpoint, planned changes for video generation.
- **Planned Changes:**
    - `app/models/manager.py`: Add imports, video model config, update `get_model`, add `generate_video` stub.
    - `app/templates/index.html`: Add 🎥 button.
    - `app/templates/gallery.html`: Add 🎥 button.
    - [x] Backend route for `/api/generate_video` (in `app/api/routes.py`).
    - [x] `GenerationPipeline.process_job` update for video jobs (in `app/models/generator.py`).
    - Frontend JS for modal and video handling.

## [2025-05-17] - Image-to-Video Model Detection Fix

- **Issue:** The application was failing to recognize "wan-i2v-14b" as a valid image-to-video model, causing an error: "Error: Model wan-i2v-14b is not a video generation model".
- **Fix:**
  - Updated `app/static/js/main.js`:
    - Modified `openVideoGenModal()` function to detect I2V models based on either:
      - Model's type property being 'i2v' OR
      - Model ID containing 'i2v' in its name
    - Added fallback description for models missing description field
  - This change allows proper detection of I2V models regardless of server-side type configuration

## [2025-05-17] - Main Form T2V/T2I Integration

- **Goal:** Allow Text-to-Video (T2V) generation directly from the main prompt form, alongside Text-to-Image (T2I).
- **Changes in `app/static/js/main.js`:**
  - Added `API_T2V_GEN` constant (`/api/generate_t2v`).
  - Modified `initializeModels()`:
    - Filters models for the main dropdown to include types 'image' and 't2v'.
    - Stores model type information.
    - Adds type label (`[Image]` or `[Video]`) to dropdown options.
  - Modified `handleModelChange()`:
    - Updates the main generate button text ("Generate Image" or "Generate Video") based on selected model type.
  - Modified `initializeGenerationForm()` submit handler:
    - Detects the selected model's type ('image' or 't2v').
    - Sets `apiUrl` to the correct endpoint (`API_IMAGE_GEN` or `API_T2V_GEN`).
    - Builds the appropriate `requestData` payload for the selected type (including `settings.type`).
    - Uses a `feedbackType` ('Image' or 'Video') for UI messages.
  - Modified `pollGenerationStatus()`:
    - Added `feedbackType` parameter to display correct messages (e.g., "Generating Video...").

## [2025-05-17] - Fix Video Export Data Type

- **Issue:** Video output (specifically from LTX-Video T2V model) was corrupt.
- **Root Cause:** The model pipeline returned frames as a Python `list`, but the `diffusers.utils.export_to_video` function expects a NumPy `ndarray`.
- **Fix:**
  - Modified `app/models/generator.py` (`process_job` method):
    - Within the T2V and I2V generation blocks, added a check before calling `export_to_video`.
    - If the `frames` variable is a `list`, convert it to a NumPy `ndarray` using `np.array()` list comprehension and `np.stack()`. Log the conversion.
    - If `frames` is already an `ndarray`, proceed as before (with existing debug logs).
    - Added error handling for the conversion process.
- **Affected Files:**
  - `app/models/generator.py`

## [2025-05-17] - Further Debugging LTX Video Export

- **Issue:** Video output still corrupt ("green lines") even after converting frame list to NumPy array.
- **Investigation:** Compared code to reference, identified potential issues:
    1.  **Dtype Mismatch:** LTX GGUF model loading in `get_model` wasn't strictly using `torch.bfloat16` as in reference.
    2.  **Frame Format:** The frames returned by `generate_text_to_video` might be Tensors, not PIL Images, causing incorrect conversion to `uint8` NumPy array in `generator.py`.
- **Fixes Applied:**
    1.  Modified `app/models/manager.py` (`get_model`): Changed LTX GGUF loading to consistently use `torch.bfloat16`.
    2.  Modified `app/models/manager.py` (`generate_text_to_video`): Added debug logging before returning `output_frames` to inspect their type, dtype, shape, and range (if tensor).
- **Next Steps:** Analyze new logs to determine frame format and adjust conversion logic in `app/models/generator.py` if necessary.
- **Affected Files:**
    - `app/models/manager.py`

## [2025-05-17] - Fix NameError in Video Debug Logging

- **Issue:** Generation failed with `NameError: name 'Image' is not defined`.
- **Root Cause:** Debug logging added in `generate_text_to_video` (`manager.py`) referenced `Image.Image` without the necessary `from PIL import Image` import.
- **Fix:** Added `from PIL import Image` at the top of `app/models/manager.py`.
- **Affected Files:**
  - `app/models/manager.py`

## [2025-05-17] - Address Video Distortion

- **Issue:** After previous fixes, video was generated but appeared distorted.
- **Investigation:** Compared application code flow to reference, identified key difference in `negative_prompt` handling.
- **Root Cause:** Application used `None` for `negative_prompt` if user provided none, while reference code used a specific quality-enhancing prompt (`"worst quality, inconsistent motion, blurry, jittery, distorted"`).
- **Fix:** Modified `generate_text_to_video` in `app/models/manager.py` to default to the reference negative prompt if none is provided in the job settings.
- **Affected Files:**
  - `app/models/manager.py`

## [2025-05-17] - Fix I2V Model Loading Error

- **Issue:** Image-to-Video generation failed with `OSError: We couldn't connect to 'https://huggingface.co' ...` and `LocalEntryNotFoundError`.
- **Root Cause:** The `wan-i2v-14b` model's components (specifically `image_encoder`) were missing from the local cache, and the loading code in `get_model` used `local_files_only=True`, preventing download.
- **Fix:** Modified the `CLIPVisionModel.from_pretrained`, `AutoencoderKLWan.from_pretrained`, and `WanImageToVideoPipeline.from_pretrained` calls within the I2V loading block in `app/models/manager.py` to use `local_files_only=False`. This allows missing components to be downloaded.
- **Affected Files:**
  - `app/models/manager.py`

## [2025-05-17] - Fix I2V Model Structure Error

- **Issue:** Image-to-Video generation failed again with `OSError: ... does not appear to have a file named pytorch_model.bin...`.
- **Investigation:** Compared model ID used by application to reference code.
- **Root Cause:** Application likely configured with incorrect repo ID (`Wan-AI/Wan2.1-I2V-14B-480P` instead of `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`). Also, VAE loading used local path instead of repo ID.
- **Fix:**
    1.  Instructed user to update `.env` to use `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` repo ID.
    2.  Modified `AutoencoderKLWan.from_pretrained` call in `app/models/manager.py` to load from the repository ID (`model_config['repo']`) instead of the local path.
- **Affected Files:**
    - `app/models/manager.py`

## [2025-05-17] - Fix I2V Job Routing

- **Issue:** I2V job processed by image generation logic, causing `TypeError` due to missing `image` argument.
- **Root Cause:** The `/generate_video` API route in `app/api/routes.py` incorrectly set the job type to `"video"` instead of `"i2v"`.
- **Fix:** Modified `/generate_video` route in `app/api/routes.py` to set `settings['type'] = 'i2v'`.
- **Affected Files:**
    - `app/api/routes.py`

## [2025-05-17] - Fix Media Deletion on Index Page

- **Issue:** Delete button failed with "NOT FOUND" error on index page for videos, while working on gallery page.
- **Root Cause:** The delete event handler in `main.js` (for index page) dynamically built the API endpoint using `mediaType` (`/api/${mediaType}/${mediaId}`), hitting non-existent `/api/video/...` DELETE endpoint. `gallery.js` correctly always used `/api/image/...`.
- **Fix:** Modified the delete event handler in `main.js` to always use the `/api/image/${mediaId}` endpoint, regardless of media type.
- **Affected Files:**
    - `app/static/js/main.js`

## [2025-05-17] - Fix I2V Out-of-Memory Error

- **Issue:** Image-to-Video generation failed with `CUDA out of memory`.
- **Root Cause:** The calculated generation resolution (e.g., 960x960) was too high for the available VRAM when using the `wan-i2v-14b` model. The reference code used a smaller `max_area`.
- **Fix:** Added `max_video_area=480*832` parameter to the `generate_image_to_video` call in `app/models/generator.py` to explicitly limit the resolution used during image preprocessing.
- **Affected Files:**
    - `app/models/generator.py`

## [2025-05-17] - Add Experimental LTX GGUF I2V Support

- **Goal:** Attempt to use the pre-loaded LTX GGUF transformer for Image-to-Video.
- **Caveat:** This deviates from LTX I2V reference code and might not be compatible.
- **Fixes:**
    1.  Modified `get_model` (`manager.py`): If `model_type` is `i2v` and source is `gguf_url`, load GGUF transformer and attempt to pass it to `LTXImageToVideoPipeline.from_pretrained`. Log warnings.
    2.  Modified `generate_image_to_video` (`manager.py`): Added branching logic based on `isinstance(pipe, LTXImageToVideoPipeline)`. For LTX, skip image resizing, use LTX-specific parameters (dimensions, frames, steps), include default negative prompt. For Wan, keep existing logic.
- **Affected Files:**
    - `app/models/manager.py`

## [2025-05-17] - Fix LTX I2V NameError

- **Issue:** LTX I2V loading failed with `NameError: name 'LTXImageToVideoPipeline' is not defined`.
- **Root Cause:** The import for `LTXImageToVideoPipeline` was missing from `app/models/manager.py`.
- **Fix:** Added `LTXImageToVideoPipeline` to the `diffusers` import statement at the top of `app/models/manager.py`.
- **Affected Files:**
    - `app/models/manager.py`

## [2025-05-17] - Simplify Video Generation Pipeline for API-Only

- **Goal:** Update the video generation pipeline to be API-only, simplifying the logic and focusing on external providers for video tasks.
- **Changes:**
    1.  **Rationale:** Fulfill user request to simplify the video pipeline to be API-only, reducing complexity and focusing on external providers for video tasks.

    2.  **File:** `app/utils/download_models.py`
        *   **Change:** Modified `download_model` function to skip downloading T2V and I2V model types.
        *   **Details:** Added a check for `model_type` being `t2v` or `i2v`. If so, logs an informational message and returns `True`, as these models are now API-only and do not require local files.
        *   **Rationale:** Align with the API-only video generation strategy, preventing unnecessary download attempts for video models.

    3.  **File:** `app/models/generator.py`
        *   **Change:** Simplified `GenerationPipeline.process_job` for T2V and I2V generation.
        *   **Details:** Removed logic that handled a list of frames and called `export_to_video` for video types. The method now expects `media_output` from `ModelManager` to always be `bytes` for video jobs. It saves these bytes directly to an .mp4 file.
        *   **Rationale:** Reflect the change that `ModelManager` provides only byte streams for API-based video generation, streamlining the processing in `generator.py`.

## [Recent Session] - T2V API Client Fixes and Token Management

- **Goal:** Address errors in Text-to-Video (T2V) generation using the Hugging Face `InferenceClient`, ensure correct API token loading, and improve parameter handling for API calls.
- **Key Modifications:**
  - **API Token Loading (`app/__init__.py`):**
    - Modified `create_app` to load `HF_TOKEN`, `FAL_AI_API_KEY`, and `REPLICATE_API_KEY` from environment variables into `current_app.config`. This makes API keys readily available for different model providers.
  - **T2V Parameter and Provider Handling (`app/models/manager.py`):**
    - In the API-based T2V generation logic (around `generate_text_to_video`):
      - Removed the unsupported `num_frames_per_second` parameter from the `InferenceClient.text_to_video()` call to prevent errors and align with the API.
      - Corrected the retrieval of the `provider` (e.g., `fal-ai`) to use `step_config.get('provider')` instead of `options.get('provider')`. This ensures the `InferenceClient` is initialized with the correct provider information.
      - Added debug logging to identify and report any unsupported parameters being passed to the API, facilitating easier debugging.
- **Affected Files:**
    - `app/__init__.py`
    - `app/models/manager.py`