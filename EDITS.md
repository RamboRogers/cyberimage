# CyberImage Code Change Log

## Change History

### [DATE] Initial Setup
- Created NOTES.md for project tracking
- Created EDITS.md for change logging
- Updated project design with specific models and storage decisions

### [DATE] Phase 1 Implementation
- Created project directory structure
- Set up requirements.txt with carefully selected dependencies
- Initialized Flask application with:
  - CORS support
  - Configuration management
  - Health check endpoint
- Created API blueprint with initial routes
- Defined available models configuration
- Created application entry point

### [DATE] Phase 2 Implementation
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

### [DATE] Phase 3 Implementation
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

### [DATE] Phase 4 Implementation
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

### [DATE] Phase 5 Implementation
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

### [DATE] Frontend Enhancements (localStorage)
- Modified `app/static/js/main.js`:
  - Added logic to `initializeModels` to load `lastModelId` from `localStorage` and apply it to the model select.
  - Added logic to model select `change` event listener to save the selected `modelId` to `localStorage`.
  - Added logic to `initializeGenerationForm` to load `lastStepsValue` from `localStorage` and apply it to the steps slider and display.
  - Added event listener to steps slider (`input`) to save its value to `localStorage`.
  - Added logic to form `submit` event listener to save both current `modelId` and `stepsValue` to `localStorage`.

### [DATE] Backend Model Step Configuration
- Updated `.env.example` and `.env`:
  - Modified format comment to include optional `<options_json>`.
  - Added `MODEL_4` entry for `sana-sprint` with `{\"fixed_steps\": 2}` configuration.
- Modified `app/utils/config.py` (`parse_model_config` function):
  - Updated logic to parse the optional 6th semicolon-separated part as a JSON string.
  - Added JSON parsing with error handling for the options string.
  - Stored the parsed dictionary (or empty dict if invalid/missing) into the `step_config` key for each model in `models_config`.
- Verified `app/api/routes.py` (`/api/models` endpoint): Confirmed it returns the `AVAILABLE_MODELS` dictionary directly, which now includes the `step_config` data. No changes needed.

### [DATE] Frontend Step Configuration Handling
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

## [YYYY-MM-DD] - Add Video Generation Feature (Initial Phase)

- **API.md:** Added documentation for new `/api/generate_video` endpoint and corresponding `GET /get_video/<video_id>`.
- **NOTES.md:** Documented design, model details, API endpoint, planned changes for video generation.
- **Planned Changes:**
    - `app/models/manager.py`: Add imports, video model config, update `get_model`, add `generate_video` stub.
    - `app/templates/index.html`: Add 🎥 button.
    - `app/templates/gallery.html`: Add 🎥 button.
    - [x] Backend route for `/api/generate_video` (in `app/api/routes.py`).
    - [x] `GenerationPipeline.process_job` update for video jobs (in `app/models/generator.py`).
    - Frontend JS for modal and video handling.

## [YYYY-MM-DD] - Image-to-Video Model Detection Fix

- **Issue:** The application was failing to recognize "wan-i2v-14b" as a valid image-to-video model, causing an error: "Error: Model wan-i2v-14b is not a video generation model".
- **Fix:**
  - Updated `app/static/js/main.js`:
    - Modified `openVideoGenModal()` function to detect I2V models based on either:
      - Model's type property being 'i2v' OR
      - Model ID containing 'i2v' in its name
    - Added fallback description for models missing description field
  - This change allows proper detection of I2V models regardless of server-side type configuration

## [YYYY-MM-DD] - Main Form T2V/T2I Integration

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