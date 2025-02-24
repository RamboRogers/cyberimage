#!/bin/bash

#if .env exists, source it
if [ -f .env ]; then
    source .env
fi
#if MODEL_FOLDER or IMAGES_FOLDER is not set, exit
if [ -z "$EXTERNAL_MODEL_FOLDER" ] || [ -z "$EXTERNAL_IMAGES_FOLDER" ]; then
    echo "❌ Error: EXTERNAL_MODEL_FOLDER or EXTERNAL_IMAGES_FOLDER is not set!"
    exit 1
fi

# Function to check if a container exists
container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^cyberimage$"
}

# Function to check if a container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^cyberimage$"
}

# Function to check if service is responding
check_service() {
    local max_attempts=30
    local attempt=1
    local wait_time=2

    echo "Checking service health..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://localhost:7860/health" | grep -q "healthy"; then
            echo "✅ Service is healthy!"
            return 0
        fi
        echo "⏳ Waiting for service to become healthy (attempt $attempt/$max_attempts)..."
        sleep $wait_time
        attempt=$((attempt + 1))
    done
    echo "❌ Service failed to become healthy after $max_attempts attempts"
    return 1
}

# Check if NVIDIA GPU is available
has_gpu() {
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Ensure .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    exit 1
fi

# Create required directories if they don't exist
mkdir -p $EXTERNAL_IMAGES_FOLDER $EXTERNAL_MODEL_FOLDER
# Ensure correct ownership
if [ "$(id -u)" -eq 0 ]; then
    chown -R 1000:1000 $EXTERNAL_IMAGES_FOLDER $EXTERNAL_MODEL_FOLDER
else
    # If not root, try sudo if the directories aren't owned by the current user
    if [ "$(stat -c '%u' $EXTERNAL_IMAGES_FOLDER)" != "$(id -u)" ]; then
        sudo chown -R "$(id -u):$(id -g)" $EXTERNAL_IMAGES_FOLDER $EXTERNAL_MODEL_FOLDER
    fi
fi

# Stop and remove existing container if it exists
if container_exists; then
    echo "🛑 Stopping existing container..."
    docker stop cyberimage
    echo "🗑️  Removing existing container..."
    docker rm cyberimage
fi

# Build the image
echo "🏗️  Building Docker image..."
docker build -t cyberimage .

# Prepare GPU options if available
GPU_OPTIONS=""
if has_gpu; then
    echo "🎮 NVIDIA GPU detected, enabling GPU support..."
    GPU_OPTIONS="--gpus all"
else
    echo "⚠️  No NVIDIA GPU detected, running in CPU mode..."
fi

# Start the container
echo "🚀 Starting container..."
docker run -d \
    --name cyberimage \
    --user $(id -u):$(id -g) \
    -p 7860:5050 \
    -v "$EXTERNAL_IMAGES_FOLDER:/app/images" \
    -v "$EXTERNAL_MODEL_FOLDER:/app/models" \
    --env-file .env \
    -e FLASK_APP=run.py \
    --restart unless-stopped \
    $GPU_OPTIONS \
    cyberimage

# Wait a moment for container to start
sleep 2

# Check if container started successfully
if ! container_running; then
    echo "❌ Failed to start container!"
    docker logs cyberimage
    exit 1
fi

# Check service health
if ! check_service; then
    echo "❌ Service health check failed!"
    docker logs cyberimage
    exit 1
fi

echo "
✨ CyberImage is ready!
📝 Logs: docker logs -f cyberimage
🔍 Status: docker ps
🏥 Health: curl http://localhost:7860/health
"