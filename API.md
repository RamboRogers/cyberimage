# CyberImage API Documentation

## Overview

CyberImage provides a RESTful API for generating high-quality images and videos using various AI models. The API supports multiple models, queued generation, and comprehensive status tracking.

## Base URL

```
http://localhost:5050/api
```

## System Architecture

CyberImage uses a single-threaded processing model for image and video generation:

- **Sequential Processing**: Only one generation job is processed at a time
- **Memory Management**: Models are loaded and unloaded between jobs to maximize available GPU memory
- **Queuing System**: Jobs are processed in order of submission (first-come, first-served)
- **Asynchronous API**: All requests are non-blocking, allowing clients to check status later

This architecture ensures stability and maximizes the quality of each generation, though it means jobs may wait in queue during high-demand periods.

## Rate Limiting

The API implements rate limiting with the following default settings:
- 10 requests per hour per IP address
- When limit is exceeded, the API returns a 429 status code with wait time information

## Model Context Protocol (MCP) Support

CyberImage now supports the [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/specification/2024-11-05/) for AI systems to interact with the image and video generation capabilities. This enables AI assistants and other systems to generate images and videos through a standardized interface.

### MCP Endpoint

The MCP endpoint is available at:

```
http://localhost:5050/api/mcp
```

### Supported MCP Methods

| Method | Description |
|--------|-------------|
| `context.image_generation.models` | List available models |
| `context.image_generation.generate` | Generate images based on a prompt |
| `context.image_generation.status` | Check generation job status |
| `context.video_generation.models` | List available video models |
| `context.video_generation.generate` | Generate videos based on a prompt |
| `context.video_generation.status` | Check video generation job status |

### MCP Request Format

All MCP requests follow the JSON-RPC 2.0 format:

```json
{
    "jsonrpc": "2.0",
    "method": "context.image_generation.METHOD_NAME",
    "params": {
        // Method parameters
    },
    "id": "request-id"
}
```

### MCP Example: Generating an Image

**Request:**

```json
{
    "jsonrpc": "2.0",
    "method": "context.image_generation.generate",
    "params": {
        "prompt": "A cyberpunk city at night with neon signs",
        "negative_prompt": "blurry, text",
        "model": "flux-2",
        "settings": {
            "num_images": 1,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024
        }
    },
    "id": "gen-123"
}
```

**Response:**

```json
{
    "jsonrpc": "2.0",
    "result": {
        "job_id": "0c907066-985e-467b-8d11-07419060ef91",
        "status": "pending",
        "num_images": 1
    },
    "id": "gen-123"
}
```

### MCP Example: Checking Job Status

**Request:**

```json
{
    "jsonrpc": "2.0",
    "method": "context.image_generation.status",
    "params": {
        "job_id": "0c907066-985e-467b-8d11-07419060ef91"
    },
    "id": "status-123"
}
```

**Response:**

```json
{
    "jsonrpc": "2.0",
    "result": {
        "job_id": "0c907066-985e-467b-8d11-07419060ef91",
        "status": "completed",
        "model": "flux-2",
        "prompt": "A cyberpunk city at night with neon signs",
        "created_at": "2024-03-20T10:30:00Z",
        "started_at": "2024-03-20T10:30:01Z",
        "completed_at": "2024-03-20T10:31:00Z",
        "progress": {
            "preparing": false,
            "loading_model": false,
            "generating": false,
            "saving": false,
            "completed": true,
            "failed": false,
            "step": 30,
            "total_steps": 30
        },
        "images": [
            {
                "id": "img_123456",
                "url": "/api/get_image/img_123456",
                "metadata": {
                    "model_id": "flux-2",
                    "prompt": "A cyberpunk city at night with neon signs",
                    "negative_prompt": null,
                    "settings": {
                        "num_inference_steps": 30,
                        "guidance_scale": 7.5,
                        "height": 1024,
                        "width": 1024
                    }
                }
            }
        ]
    },
    "id": "status-123"
}
```

### MCP Example Client

A Python example client for the MCP endpoint is included in the `examples/mcp_client_example.py` file.

## RESTful API Endpoints

The following RESTful API endpoints are available in addition to the MCP endpoint:

### 1. List Available Models
Get a list of all available models and their configurations.

**Endpoint:** `GET /models`

**Response:**
```json
{
    "models": {
        "flux-2": {
            "id": "black-forest-labs/FLUX.1-dev",
            "description": "FP8 quantized version of FLUX-1 for memory efficiency"
        },
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
    "default": "flux-2"
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
        "num_images": 1,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "height": 1024,
        "width": 1024,
        "max_sequence_length": 512
    }
}
```

**Parameters:**
- `model_id` (required): One of the available model IDs ("flux-1", "flux-2", "sd-3.5", "flux-abliterated")
- `prompt` (required): Text description of the image to generate (max 500 characters)
- `negative_prompt` (optional): Text description of what to avoid in the image (max 500 characters)
  - Note: Negative prompts are only supported by SD 3.5 model
  - For Flux models, this parameter will be gracefully ignored
- `settings` (optional): Generation parameters
  - `num_images`: Number of images to generate per job (default: 1, max: 8)
  - `num_inference_steps`: Number of denoising steps (default: 30)
  - `guidance_scale`: How closely to follow the prompt (default: 7.5)
  - `height`: Image height in pixels (default: 1024)
  - `width`: Image width in pixels (default: 1024)
  - `max_sequence_length`: Maximum prompt length (default: 512)

**Model-Specific Behavior:**
- Flux models don't support negative prompts (parameter is ignored)
- All models support the same standard generation parameters
- Progress tracking is done at the job level, not with individual model steps

**Response:**
```json
{
    "job_id": "0c907066-985e-467b-8d11-07419060ef91",
    "status": "pending",
    "num_images": 1,
    "message": "Image generation job submitted successfully for 1 image(s)"
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
        "num_images": 2,
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
    "status": "processing",
    "model_id": "flux-1",
    "prompt": "A cyberpunk city at night...",
    "negative_prompt": "blurry, text",
    "settings": {
        "num_inference_steps": 35,
        "guidance_scale": 8.0,
        "height": 1024,
        "width": 1024
    },
    "created_at": "2024-03-20T10:30:00Z",
    "started_at": "2024-03-20T10:30:01Z",
    "completed_at": null,
    "message": "Generating image 1 of 2... Step 15/30",
    "progress": {
        "preparing": false,
        "loading_model": false,
        "generating": true,
        "saving": false,
        "completed": false,
        "failed": false,
        "step": 15,
        "total_steps": 30
    },
    "images": []
}
```

When the job is completed, the response includes image information:

```json
{
    "id": "0c907066-985e-467b-8d11-07419060ef91",
    "status": "completed",
    "model_id": "flux-1",
    "prompt": "A cyberpunk city at night...",
    "negative_prompt": "blurry, text",
    "settings": {
        "num_inference_steps": 35,
        "guidance_scale": 8.0,
        "height": 1024,
        "width": 1024
    },
    "created_at": "2024-03-20T10:30:00Z",
    "started_at": "2024-03-20T10:30:01Z",
    "completed_at": "2024-03-20T10:31:00Z",
    "message": "Generated 2 image(s)",
    "progress": {
        "preparing": false,
        "loading_model": false,
        "generating": false,
        "saving": false,
        "completed": true,
        "failed": false,
        "step": 30,
        "total_steps": 30
    },
    "images": [
        {
            "id": "img_123456",
            "file_path": "2024/03/20/img_123456.png",
            "created_at": "2024-03-20T10:30:45",
            "metadata": {
                "model_id": "flux-1",
                "prompt": "A cyberpunk city at night...",
                "negative_prompt": null,
                "settings": {
                    "num_inference_steps": 35,
                    "guidance_scale": 8.0,
                    "height": 1024,
                    "width": 1024
                },
                "image_number": 1,
                "total_images": 2
            }
        },
        {
            "id": "img_123457",
            "file_path": "2024/03/20/img_123457.png",
            "created_at": "2024-03-20T10:31:00",
            "metadata": {
                "model_id": "flux-1",
                "prompt": "A cyberpunk city at night...",
                "negative_prompt": null,
                "settings": {
                    "num_inference_steps": 35,
                    "guidance_scale": 8.0,
                    "height": 1024,
                    "width": 1024
                },
                "image_number": 2,
                "total_images": 2
            }
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
    "model_id": "flux-1",
    "prompt": "A cyberpunk city at night...",
    "negative_prompt": null,
    "settings": {
        "num_inference_steps": 35,
        "guidance_scale": 8.0,
        "height": 1024,
        "width": 1024
    },
    "generation_time": 1616257845.123,
    "image_number": 1,
    "total_images": 2
}
```

**Example curl:**
```bash
curl -X GET http://localhost:5050/api/image/img_123456/metadata
```

### 6. Delete Image
Delete a generated image.

**Endpoint:** `DELETE /image/<image_id>`

**Response:**
```json
{
    "status": "success",
    "message": "Image deleted successfully"
}
```

**Example curl:**
```bash
curl -X DELETE http://localhost:5050/api/image/img_123456
```

### 7. View Queue Status
Get the current status of the generation queue.

**Endpoint:** `GET /queue`

**Parameters:**
- `detailed` (optional): Set to "true" to get enhanced statistics (default: false)

**Basic Response:**
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

**Detailed Response (with ?detailed=true):**
```json
{
    "pending": 2,
    "processing": 1,
    "completed": 150,
    "failed": 3,
    "total": 156,
    "queue_size": 2,
    "recent_24h": {
        "pending": 1,
        "processing": 1,
        "completed": 25,
        "failed": 2,
        "total": 29
    },
    "avg_processing_time_seconds": 45.23,
    "models": {
        "flux-1": {
            "pending": 1,
            "processing": 0,
            "completed": 10,
            "failed": 1,
            "total": 12
        },
        "sd-3.5": {
            "pending": 0,
            "processing": 1,
            "completed": 15,
            "failed": 1,
            "total": 17
        }
    },
    "failure_rate": 7.41
}
```

**Example curl:**
```bash
curl -X GET http://localhost:5050/api/queue
```

**Example curl (detailed):**
```bash
curl -X GET "http://localhost:5050/api/queue?detailed=true"
```

### 8. Reset Stalled Jobs
Reset any stalled jobs back to pending status.

**Endpoint:** `POST /reset_queue`

**Response:**
```json
{
    "status": "success",
    "reset_count": 2,
    "message": "Reset 2 stalled jobs to pending status"
}
```

**Example curl:**
```bash
curl -X POST http://localhost:5050/api/reset_queue
```

### 9. Get Gallery Images
Get paginated gallery of generated images with search capabilities.

**Endpoint:** `GET /gallery`

**Parameters:**
- `page` (optional): Page number (default: 1)
- `limit` (optional): Number of images per page (default: 20, max: 50)
- `search` (optional): Search query to filter images by prompt or model
- `model` (optional): Filter by specific model ID

**Response:**
```json
{
    "images": [
        {
            "id": "img_123456",
            "model_id": "flux-1",
            "prompt": "A cyberpunk city at night...",
            "created_at": "2024-03-20T10:31:00Z",
            "settings": {
                "num_inference_steps": 35,
                "guidance_scale": 8.0
            }
        },
        {
            "id": "img_123457",
            "model_id": "sd-3.5",
            "prompt": "A fantasy landscape with mountains...",
            "created_at": "2024-03-20T10:15:00Z",
            "settings": {
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            }
        }
    ],
    "page": 1,
    "limit": 20,
    "has_more": true,
    "total": 156
}
```

**Example curl:**
```bash
curl -X GET "http://localhost:5050/api/gallery?page=1&limit=20"
```

**Example curl (with search):**
```bash
curl -X GET "http://localhost:5050/api/gallery?search=cyberpunk&model=flux-1"
```

### 10. Enrich Prompt
Enhance a basic prompt with additional details based on a selected style.

**Endpoint:** `POST /enrich`

**Request Body:**
```json
{
    "prompt": "a castle on a hill",
    "style": "fantasy"
}
```

**Parameters:**
- `prompt` (required): The basic prompt to enhance
- `style` (optional): Style template to use (default: "cyberpunk")
  - Available styles: "cyberpunk", "anime", "realistic", "fantasy", "sci-fi", "enhance"

**Response:**
```json
{
    "original_prompt": "a castle on a hill",
    "enriched_prompt": "a majestic fantasy castle perched atop a rolling hill, ancient stone architecture with ornate towers and flying buttresses, lush magical landscape surrounding the fortress, ethereal mist rising from the valley below, dramatic sky with scattered clouds, golden hour lighting, richly detailed textures, digital painting, concept art, 8k resolution",
    "style": "fantasy"
}
```

**Example curl:**
```bash
curl -X POST http://localhost:5050/api/enrich \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a castle on a hill",
    "style": "fantasy"
}'
```

### 11. Generate Text-to-Video (New)
Submit a new video generation request based purely on text input.

**Endpoint:** `POST /generate_t2v`

**Request Body:**
```json
{
    "model_id": "wan-t2v", // Example ID for a Text-to-Video model
    "prompt": "A fluffy cat chasing a laser pointer across a wooden floor",
    "settings": {
        "guidance_scale": 5.5,
        "fps": 16,
        "duration": 8 // Desired duration in seconds (example)
        // Other video-specific settings might go here
    }
}
```

**Parameters:**
- `model_id` (required): The ID of the Text-to-Video generation model to use (needs to be listed in `/models` response, marked appropriately).
- `prompt` (required): Text description guiding the video generation.
- `settings` (optional): Video generation parameters.
  - `guidance_scale`: How closely to follow the prompt (default might vary by model).
  - `fps`: Frames per second for the output video (default: 16).
  - `duration`: Desired duration of the video in seconds (model/implementation dependent).
  - Note: Height/Width might be fixed by the model or configurable via settings.

**Response (Job Submission):**
```json
{
    "job_id": "t2vjob_abcdef123",
    "status": "pending",
    "message": "Text-to-Video generation job submitted successfully."
}
```

**Response (Job Status `GET /status/<job_id>`):**
Similar to image/video generation, but status updates reflect text-to-video progress. Completed status will include video details. The `videos` array will contain the output.

**Get Generated Video:** Use the existing `GET /get_video/<video_id>` endpoint.

**Example curl (Generate T2V):**
```bash
curl -X POST http://localhost:5050/api/generate_t2v \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_id": "wan-t2v",
    "prompt": "A fluffy cat chasing a laser pointer across a wooden floor",
    "settings": { "fps": 24, "duration": 10 }
  }'
```

### 12. Generate Image-to-Video (Previously Generate Video)
Submit a new video generation request based on an existing image and a guiding prompt.

**Endpoint:** `POST /generate_video`

**Request Body:**
```json
{
    "source_image_id": "img_123456",
    "video_model_id": "Wan-AI/Wan2.1-I2V-14B-480P", // Example ID for an Image-to-Video model
    "video_prompt": "The camera slowly zooms out, revealing the bird is on a branch...",
    "settings": {
        "guidance_scale": 5.5,
        "fps": 16
        // Other video-specific settings might go here
    }
}
```

**Parameters:**
- `source_image_id` (required): The ID of the existing generated image to use as the starting point.
- `video_model_id` (required): The ID of the Image-to-Video generation model to use (needs to be listed in `/models` response, marked appropriately).
- `video_prompt` (required): Text description guiding the video generation (motion, changes).
- `settings` (optional): Video generation parameters.
  - `guidance_scale`: How closely to follow the prompt (default might vary by model).
  - `fps`: Frames per second for the output video (default: 16).
  - Note: Height/Width will be derived from the source image and model constraints.

**Response (Job Submission):**
```json
{
    "job_id": "i2vjob_abcdef123",
    "status": "pending",
    "message": "Image-to-Video generation job submitted successfully."
}
```

**Response (Job Status `GET /status/<job_id>`):**
Similar to other generation jobs. Completed status will include video details in the `videos` array.

```json
{
    "id": "i2vjob_abcdef123",
    "status": "completed",
    "model_id": "Wan-AI/Wan2.1-I2V-14B-480P",
    "video_prompt": "The camera slowly zooms out...",
    "source_image_id": "img_123456",
    "settings": { // ... video settings ... },
    // ... timestamps ...
    "message": "Generated 1 video",
    "progress": { // ... progress details ... },
    "videos": [ // Changed from "images" for clarity
        {
            "id": "vid_78910", // New ID for the video record
            "file_path": "videos/2024/07/26/vid_78910.mp4", // Example path
            "created_at": "2024-07-26T14:00:00Z",
            "metadata": {
                "video_model_id": "Wan-AI/Wan2.1-I2V-14B-480P",
                "video_prompt": "The camera slowly zooms out...",
                "source_image_id": "img_123456",
                "settings": { // ... video settings ... },
                "duration_seconds": 5 // Example duration
            }
        }
    ]
}
```

**Get Generated Video:** Use the existing `GET /get_video/<video_id>` endpoint.

**Endpoint:** `GET /get_video/<video_id>`

**Response:** MP4 video file

**Example curl (Generate I2V):**
```bash
curl -X POST http://localhost:5050/api/generate_video \\
  -H "Content-Type: application/json" \\
  -d '{
    "source_image_id": "img_123456",
    "video_model_id": "Wan-AI/Wan2.1-I2V-14B-480P",
    "video_prompt": "The scene animates with subtle wind blowing through the trees",
    "settings": { "guidance_scale": 5.0 }
  }'
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
        "num_images": 2,
        "num_inference_steps": 35,
        "guidance_scale": 8.0
    }
})
job_id = response.json()["job_id"]

# 3. Poll for completion
while True:
    status = requests.get(f"{API_BASE}/status/{job_id}").json()

    # Display progress information
    if status["status"] == "processing" and "progress" in status:
        progress = status["progress"]
        if progress["generating"] and progress["step"] is not None:
            print(f"Generating: Step {progress['step']}/{progress['total_steps']}")

    if status["status"] == "completed":
        # 4. Download all generated images
        for i, image_info in enumerate(status["images"]):
            image_id = image_info["id"]
            image = requests.get(f"{API_BASE}/get_image/{image_id}")
            with open(f"generated_image_{i+1}.png", "wb") as f:
                f.write(image.content)
            print(f"Saved image {i+1}/{len(status['images'])}")
        break
    elif status["status"] == "failed":
        print(f"Generation failed: {status.get('message')}")
        break

    time.sleep(5)
```