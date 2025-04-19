# CyberImage Project Notes

## Project State
- Initial setup phase
- Planning core architecture

## Core Components Tracking
### Storage Structure
- /images - Local image storage
- /db - SQLite database storage

### API Endpoints (REST)
- POST /api/generate - Submit generation request
- GET /api/status/{job_id} - Check job status
- GET /api/image/{image_id} - Retrieve generated image
- GET /api/models - List available models
- GET /api/queue - View current queue status

### Database Schema
#### Jobs Table
```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    status TEXT,  -- PENDING, PROCESSING, COMPLETED, FAILED
    model TEXT,
    prompt TEXT,
    created_at TIMESTAMP,
    completed_at TIMESTAMP,
    image_path TEXT,
    settings JSON
);
```

### Models Integration
Available Models:
1. FLUX.1-dev (black-forest-labs/FLUX.1-dev)
2. SD 3.5 Large (stabilityai/stable-diffusion-3.5-large)
3. Flux.1dev Abliterated (aoxo/flux.1dev-abliteratedv2)

#### Proposed .env Format Update (Model Configuration)
Current Format: `MODEL_NAME=<name>;<repo>;<description>;<source>;<requires_auth>`
Proposed Format: `MODEL_NAME=<name>;<repo>;<description>;<source>;<requires_auth>[;<options_json>]`

The `<options_json>` is an optional JSON string to specify model-specific behaviors, such as step configurations.

Examples:
- **Fixed Steps:** `MODEL_X="fixed-step-model;some/repo;Fixed Step Model;huggingface;false;{\\"fixed_steps\\": 25}"`
- **Step Range:** `MODEL_Y="range-step-model;another/repo;Range Step Model;huggingface;false;{\\"steps\\": {\\"min\\": 10, \\"max\\": 40, \\"default\\": 20}}"`
- **Use Case:** `MODEL_SANA="sana-sprint;Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers;Sana Sprint 1.6B;huggingface;false;{\\"fixed_steps\\": 2}"`

*Backend changes needed to parse this and update the `/api/models` endpoint.*

### Core Functions
```python
def generate_image(prompt: str, model_id: str) -> str:
    """Generate image and return job_id"""

def get_job_status(job_id: str) -> dict:
    """Get status of generation job"""

def get_image(image_id: str) -> str:
    """Get path to generated image"""

def process_queue() -> None:
    """Process pending generation jobs"""
```

## Frontend Enhancements (main.js)

### Completed: User Settings Persistence
- **Goal:** Remember user's last selected Model and Steps value between sessions.
- **Implementation:** Used `localStorage` in `app/static/js/main.js`.
    - On page load (`DOMContentLoaded`), retrieve `lastModelId` and `lastStepsValue` from `localStorage` and apply to form elements (`#model`, `#steps`).
    - On `#model` change, save new `modelId` to `localStorage`.
    - On `#steps` input change, save new `stepsValue` to `localStorage`.
    - On form submission (`#generate-form`), save current `modelId` and `stepsValue` to `localStorage`.

### Planned: Model-Specific Step Handling
- **Goal:** Adapt the Steps slider UI based on the selected model's configuration (fixed steps, range, or default).
- **Prerequisite:** Backend update to parse new `.env` format and include step info in `/api/models` response.
- **Implementation Plan (`main.js`):**
    - In the `#model` select `change` event listener:
        - Get step configuration data for the selected model from the data fetched during `initializeModels`.
        - If `fixed_steps` is defined:
            - Set the `#steps` slider value to `fixed_steps`.
            - Disable the `#steps` slider.
            - Update the `#steps-value` display.
        - If `steps` range (`min`, `max`, `default`) is defined:
            - Update `#steps` slider `min`, `max`, `value` attributes.
            - Enable the `#steps` slider.
            - Update the `#steps-value` display.
        - Otherwise (no specific step info):
            - Reset `#steps` slider to default range/value (e.g., min 20, max 50, value 30).
            - Enable the `#steps` slider.
            - Update the `#steps-value` display.

## Technical Decisions
- Python 3.12 + Flask Backend
- SQLite for data persistence and queue management
- Local filesystem for image storage
- REST API only, no WebSocket
- No authentication required
- Models loaded on-demand to manage memory

## Progress Log
[Current] Planning Phase - Initial Architecture Design
- Defined storage structure
- Specified database schema
- Identified core API endpoints
- Listed supported models