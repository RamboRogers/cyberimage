# CyberImage API Documentation

## Overview

CyberImage provides a RESTful API for generating high-quality images using various AI models. The API supports multiple models, queued generation, and comprehensive status tracking.

## Base URL

```
http://localhost:5050/api
```

## Rate Limiting

The API implements rate limiting with the following default settings:
- 10 requests per hour per IP address
- When limit is exceeded, the API returns a 429 status code with wait time information

## Available Endpoints

### 1. List Available Models
Get a list of all available models and their configurations.

**Endpoint:** `GET /models`

**Response:**
```json
{
    "models": {
        "flux-1": {
            "id": "black-forest-labs/FLUX.1-dev",
            "description": "FLUX base model"
        },
        "sd-3.5": {
            "id": "stabilityai/stable-diffusion-3.5-large",
            "description": "Latest Stable Diffusion model with improved quality and speed"
        },
        "flux-abliterated": {
            "id": "aoxo/flux.1dev-abliteratedv2",
            "description": "FLUX Abliterated variant"
        }
    },
    "default": "flux-1"
}
```

**Example curl:**
```bash
curl -X GET http://localhost:5050/api/models
```

### 2. Generate Image
Submit a new image generation request.

**Endpoint:** `POST /generate`

**Request Body:**
```json
{
    "model_id": "flux-1",
    "prompt": "Your detailed image description",
    "negative_prompt": "Optional text describing what you don't want in the image",
    "settings": {
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "height": 1024,
        "width": 1024,
        "max_sequence_length": 512
    }
}
```

**Parameters:**
- `model_id` (required): One of the available model IDs ("flux-1", "sd-3.5", "flux-abliterated")
- `prompt` (required): Text description of the image to generate (max 500 characters)
- `negative_prompt` (optional): Text description of what to avoid in the image (max 500 characters)
  - Note: Negative prompts are only supported by certain models (e.g., SD 3.5)
  - For unsupported models, this parameter will be gracefully ignored
- `settings` (optional): Generation parameters
  - `num_inference_steps`: Number of denoising steps (default: 30)
  - `guidance_scale`: How closely to follow the prompt (default: 7.5)
  - `height`: Image height in pixels (default: 1024)
  - `width`: Image width in pixels (default: 1024)
  - `max_sequence_length`: Maximum prompt length (default: 512)

**Response:**
```json
{
    "job_id": "0c907066-985e-467b-8d11-07419060ef91",
    "status": "pending",
    "message": "Image generation job submitted successfully"
}
```

**Example curl:**
```bash
curl -X POST http://localhost:5050/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "sd-3.5",
    "prompt": "A cyberpunk city at night with neon signs and flying cars, highly detailed",
    "negative_prompt": "blurry, low quality, distorted, bad anatomy, text, watermark",
    "settings": {
        "num_inference_steps": 35,
        "guidance_scale": 8.0,
        "height": 1024,
        "width": 1024
    }
}'
```

### 3. Check Generation Status
Get the current status of a generation job.

**Endpoint:** `GET /status/<job_id>`

**Response:**
```json
{
    "id": "0c907066-985e-467b-8d11-07419060ef91",
    "status": "completed",
    "model_id": "flux-1",
    "prompt": "A cyberpunk city at night...",
    "settings": {
        "num_inference_steps": 35,
        "guidance_scale": 8.0,
        "height": 1024,
        "width": 1024
    },
    "created_at": "2024-03-20T10:30:00Z",
    "started_at": "2024-03-20T10:30:01Z",
    "completed_at": "2024-03-20T10:31:00Z",
    "images": [
        {
            "id": "img_123456",
            "url": "/api/get_image/img_123456"
        }
    ]
}
```

**Status Values:**
- `pending`: Job is queued
- `processing`: Job is currently generating
- `completed`: Job finished successfully
- `failed`: Job failed (includes error_message)

**Example curl:**
```bash
curl -X GET http://localhost:5050/api/status/0c907066-985e-467b-8d11-07419060ef91
```

### 4. Get Generated Image
Download a generated image.

**Endpoint:** `GET /get_image/<image_id>`

**Response:** PNG image file

**Example curl:**
```bash
curl -X GET http://localhost:5050/api/get_image/img_123456 --output image.png
```

### 5. Get Image Metadata
Get metadata for a generated image.

**Endpoint:** `GET /image/<image_id>/metadata`

**Response:**
```json
{
    "id": "img_123456",
    "job_id": "0c907066-985e-467b-8d11-07419060ef91",
    "model_id": "flux-1",
    "prompt": "A cyberpunk city at night...",
    "settings": {
        "num_inference_steps": 35,
        "guidance_scale": 8.0,
        "height": 1024,
        "width": 1024
    },
    "created_at": "2024-03-20T10:31:00Z"
}
```

**Example curl:**
```bash
curl -X GET http://localhost:5050/api/image/img_123456/metadata
```

### 6. View Queue Status
Get the current status of the generation queue.

**Endpoint:** `GET /queue`

**Response:**
```json
{
    "pending": 2,
    "processing": 1,
    "completed": 150,
    "failed": 3,
    "total": 156,
    "queue_size": 2
}
```

**Example curl:**
```bash
curl -X GET http://localhost:5050/api/queue
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- 200: Success
- 400: Bad Request (invalid parameters)
- 404: Not Found (invalid job_id or image_id)
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error

Error Response Format:
```json
{
    "status": "error",
    "message": "Error description"
}
```

## Building a Client

When building a client application, follow this typical workflow:

1. Check available models using `GET /models`
2. Submit generation request using `POST /generate`
3. Poll job status using `GET /status/<job_id>` (recommended interval: 5 seconds)
4. Once status is "completed", fetch image using `GET /get_image/<image_id>`
5. Optionally fetch metadata using `GET /image/<image_id>/metadata`
6. Monitor queue status using `GET /queue`

### Example Client Workflow:

```python
import requests
import time

API_BASE = "http://localhost:5050/api"

# 1. Get available models
models = requests.get(f"{API_BASE}/models").json()

# 2. Submit generation request
response = requests.post(f"{API_BASE}/generate", json={
    "model_id": "flux-1",
    "prompt": "A cyberpunk city at night with neon signs and flying cars",
    "settings": {
        "num_inference_steps": 35,
        "guidance_scale": 8.0
    }
})
job_id = response.json()["job_id"]

# 3. Poll for completion
while True:
    status = requests.get(f"{API_BASE}/status/{job_id}").json()
    if status["status"] == "completed":
        # 4. Download image
        image_id = status["images"][0]["id"]
        image = requests.get(f"{API_BASE}/get_image/{image_id}")
        with open("generated_image.png", "wb") as f:
            f.write(image.content)
        break
    elif status["status"] == "failed":
        print(f"Generation failed: {status.get('error_message')}")
        break
    time.sleep(5)
```