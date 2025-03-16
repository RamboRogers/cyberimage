# Integrating CyberImage with AI Systems using MCP

This guide explains how to integrate CyberImage with AI assistants and other systems using the Model Context Protocol (MCP).

## What is Model Context Protocol?

The [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/specification/2024-11-05/) is an open protocol that standardizes how AI systems interact with external tools and capabilities. It allows AI models to access context from various sources and use external tools through a standardized JSON-RPC interface.

## CyberImage MCP Integration

CyberImage implements MCP to enable AI assistants to generate images directly using our image generation capabilities. The MCP integration uses the same default model as the web interface, ensuring consistency across all application interfaces. This default model is determined dynamically, prioritizing "flux-1" if available, or otherwise using the first available model in the configuration.

This provides a standardized way for AI systems to:

1. Discover available image generation models
2. Generate images based on text prompts
3. Track the progress of image generation
4. Retrieve the resulting images

## MCP Endpoint

The MCP endpoint is accessible at:

```
http://localhost:5050/api/mcp
```

For production deployments, replace `localhost:5050` with your actual server address.

## Supported Methods

CyberImage supports the following MCP methods:

### 1. context.image_generation.models

Lists all available image generation models.

**Request:**
```json
{
    "jsonrpc": "2.0",
    "method": "context.image_generation.models",
    "id": "request-123"
}
```

**Response:**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "models": {
            "flux-2": {
                "id": "black-forest-labs/FLUX.1-dev",
                "description": "FP8 quantized version of FLUX-1 for memory efficiency"
            },
            "flux-1": {
                "id": "black-forest-labs/FLUX.1-dev",
                "description": "High-quality image generation model optimized for detailed outputs"
            },
            "sd-3.5": {
                "id": "stabilityai/stable-diffusion-3.5-large",
                "description": "Latest Stable Diffusion model with improved quality and speed"
            },
            "flux-abliterated": {
                "id": "aoxo/flux.1dev-abliteratedv2",
                "description": "Modified FLUX model with enhanced capabilities"
            }
        },
        "default": "flux-1"
    },
    "id": "request-123"
}
```

The `default` field in the response indicates the system's current default model, which is dynamically determined based on available models (prioritizing "flux-1" if available, otherwise using the first available model).

### 2. context.image_generation.generate

Submits an image generation request.

**Request:**
```json
{
    "jsonrpc": "2.0",
    "method": "context.image_generation.generate",
    "params": {
        "prompt": "A detailed description of the image you want to generate",
        "negative_prompt": "What to exclude from the image (optional)",
        "model": "flux-1",
        "settings": {
            "num_images": 1,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "height": 1024,
            "width": 1024
        }
    },
    "id": "request-123"
}
```

If the `model` parameter is omitted, the system will use the default model as returned by the `context.image_generation.models` method.

**Response:**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "job_id": "0c907066-985e-467b-8d11-07419060ef91",
        "status": "pending",
        "num_images": 1
    },
    "id": "request-123"
}
```

### 3. context.image_generation.status

Checks the status of a generation job.

**Request:**
```json
{
    "jsonrpc": "2.0",
    "method": "context.image_generation.status",
    "params": {
        "job_id": "0c907066-985e-467b-8d11-07419060ef91"
    },
    "id": "request-123"
}
```

**Response (when pending/processing):**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "job_id": "0c907066-985e-467b-8d11-07419060ef91",
        "status": "processing",
        "model": "flux-2",
        "prompt": "A detailed description...",
        "created_at": "2024-03-20T10:30:00Z",
        "started_at": "2024-03-20T10:30:01Z",
        "completed_at": null,
        "progress": {
            "preparing": false,
            "loading_model": false,
            "generating": true,
            "saving": false,
            "completed": false,
            "failed": false,
            "step": 15,
            "total_steps": 30
        }
    },
    "id": "request-123"
}
```

**Response (when completed):**
```json
{
    "jsonrpc": "2.0",
    "result": {
        "job_id": "0c907066-985e-467b-8d11-07419060ef91",
        "status": "completed",
        "model": "flux-2",
        "prompt": "A detailed description...",
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
                    "prompt": "A detailed description...",
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
    "id": "request-123"
}
```

## Typical Integration Flow

1. **Model Discovery**: Call `context.image_generation.models` to discover available models.
2. **Image Generation**: Call `context.image_generation.generate` with a prompt and optionally a model selection.
3. **Status Polling**: Call `context.image_generation.status` periodically to check job progress.
4. **Image Retrieval**: Once the job is completed, use the URL from the response to retrieve the image.

## Integration Examples

### Python Example (Basic)

```python
import requests
import json
import time

# Configuration
MCP_ENDPOINT = "http://localhost:5050/api/mcp"

def mcp_request(method, params=None):
    """Make a request to the MCP endpoint"""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": "request-" + str(int(time.time()))
    }

    response = requests.post(MCP_ENDPOINT, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

def get_default_model():
    """Get the default model from the system"""
    response = mcp_request("context.image_generation.models")
    if response and "result" in response:
        return response["result"]["default"]
    return None

def generate_image(prompt, model=None, negative_prompt=None):
    """Generate an image using the MCP endpoint"""
    # Prepare parameters
    params = {"prompt": prompt}
    if model:
        params["model"] = model
    if negative_prompt:
        params["negative_prompt"] = negative_prompt

    # Submit job
    response = mcp_request("context.image_generation.generate", params)
    if not response or "error" in response:
        print("Error generating image")
        return None

    job_id = response["result"]["job_id"]
    print(f"Job submitted with ID: {job_id}")

    # Poll for completion
    while True:
        time.sleep(3)  # Wait between polls
        status_response = mcp_request("context.image_generation.status", {"job_id": job_id})

        if not status_response or "error" in status_response:
            print("Error checking status")
            break

        status = status_response["result"]["status"]
        print(f"Status: {status}")

        if status == "completed":
            # Get the image URL
            image_url = status_response["result"]["images"][0]["url"]
            full_url = f"http://localhost:5050{image_url}" if image_url.startswith("/") else image_url
            print(f"Image available at: {full_url}")
            return full_url

        elif status == "failed":
            print("Generation failed")
            break

    return None

# Example usage
if __name__ == "__main__":
    # First example: Using the system's default model
    default_model = get_default_model()
    print(f"Using system default model: {default_model}")

    image_url = generate_image("A beautiful sunset over mountains")
    if image_url:
        print(f"Success! Image available at: {image_url}")

    # Second example: Explicitly specifying a model
    image_url = generate_image("A futuristic cityscape", model="sd-3.5")
    if image_url:
        print(f"Success! Image available at: {image_url}")
```

### Complete AI Assistant Integration

For a complete AI assistant integration example, see the `examples/ai_assistant_mcp_example.py` file in the repository.

## Best Practices

1. **Handle Rate Limiting**: Implement exponential backoff when making repeated requests.
2. **Provide Detailed Prompts**: The quality of the generated image depends on the prompt detail.
3. **Use Model Discovery**: Call the `context.image_generation.models` method to discover available models and the current default model.
4. **Error Handling**: Be prepared to handle error responses from the API.
5. **Timeouts**: Set appropriate timeouts for both the initial request and status polling.
6. **User Consent**: When using in AI assistants, ensure users are aware of and consent to image generation.

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure the CyberImage server is running and accessible.
2. **Invalid Method**: Check that you're using the correct method names as listed above.
3. **Job Not Found**: The job ID might be invalid or the job may have been cleaned up.
4. **Rate Limiting**: You may be making too many requests in a short period.

### Debugging

For debugging, you can set the `DEBUG` environment variable to see more detailed logs:

```bash
DEBUG=1 python your_integration_script.py
```

## Getting Help

If you encounter issues with MCP integration, please:

1. Check the examples in the `examples/` directory
2. Review the API documentation in `API.md`
3. Open an issue on the GitHub repository