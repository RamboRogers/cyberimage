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