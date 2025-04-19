# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and newer libstdc++
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get update \
    && apt-get install -y gcc-12 g++-12 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Create non-root user
RUN useradd -u 1000 -m -s /bin/bash appuser \
    && chown -R appuser:appuser /app

RUN pip install --no-cache-dir uv

# Install Python dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt --system

# Copy application code
COPY --chown=appuser:appuser . .
ENV FLASK_APP=run.py

# Create mount points and set permissions
RUN mkdir -p /app/images /app/models \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5050

# Run with Gunicorn in production mode - ensure single thread processing
CMD ["gunicorn", "--workers=1", "--threads=1", "--bind=0.0.0.0:5050", "--timeout=120", "run:app"]