#!/usr/bin/env python3
"""
GPU Watchdog Test Script for CyberImage

This script allows testing the GPU watchdog and memory management functionality
by simulating various scenarios including memory leaks, stalled jobs, and recovery.
"""
import os
import sys
import time
import json
import argparse
import requests
import threading
from tqdm import tqdm
import torch

# Base API URL
API_BASE = "http://localhost:5050/api"

# Test prompts
TEST_PROMPTS = [
    "A cyberpunk city at night with neon signs and flying cars, highly detailed",
    "A futuristic metropolis with towering skyscrapers and holographic billboards",
    "A mystical forest with glowing mushrooms and floating crystals, fantasy art",
    "A robot samurai standing in a Zen garden, intricate digital art",
    "An abandoned space station floating in orbit around an alien planet",
    "A post-apocalyptic landscape with overgrown ruins and a vibrant sunset",
]

# Output formatting
def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title.upper()} ".center(80, '='))
    print("=" * 80)

def print_status(message, status="info"):
    """Print formatted status messages"""
    status_icons = {
        "info": "ℹ️",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "pending": "⏳"
    }
    icon = status_icons.get(status, "ℹ️")
    print(f"\n{icon} {message}")
    sys.stdout.flush()

def check_gpu_memory():
    """Check current GPU memory usage"""
    if not torch.cuda.is_available():
        print_status("CUDA not available, skipping memory check", "warning")
        return None

    try:
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print_status(f"GPU Memory Status:", "info")
        print(f"   • Total Memory: {total_memory:.2f}GB")
        print(f"   • Allocated Memory: {memory_allocated:.2f}GB")
        print(f"   • Reserved Memory: {memory_reserved:.2f}GB")
        print(f"   • Usage: {memory_allocated/total_memory:.2%}")

        return {
            "total": total_memory,
            "allocated": memory_allocated,
            "reserved": memory_reserved,
            "usage_percentage": memory_allocated/total_memory
        }
    except Exception as e:
        print_status(f"Error checking GPU memory: {str(e)}", "error")
        return None

def submit_job(model_id="flux-1", prompt=None):
    """Submit a generation job"""
    if prompt is None:
        import random
        prompt = random.choice(TEST_PROMPTS)

    try:
        print_status(f"Submitting job for model: {model_id}", "info")
        print(f"   • Prompt: {prompt}")

        response = requests.post(f"{API_BASE}/generate", json={
            "model_id": model_id,
            "prompt": prompt,
            "settings": {
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "height": 1024,
                "width": 1024
            }
        })

        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data["job_id"]
            print_status(f"Job submitted successfully: {job_id}", "success")
            return job_id
        else:
            print_status(f"Failed to submit job. Status: {response.status_code}, Response: {response.text}", "error")
            return None
    except Exception as e:
        print_status(f"Error submitting job: {str(e)}", "error")
        return None

def check_job_status(job_id):
    """Check the status of a submitted job"""
    try:
        response = requests.get(f"{API_BASE}/status/{job_id}")
        if response.status_code == 200:
            status_data = response.json()
            return status_data
        else:
            print_status(f"Failed to check job status. Status: {response.status_code}", "error")
            return None
    except Exception as e:
        print_status(f"Error checking job status: {str(e)}", "error")
        return None

def wait_for_job(job_id, timeout=300, check_interval=2):
    """Wait for a job to complete with progress tracking"""
    start_time = time.time()
    previous_status = None

    with tqdm(total=100, desc="Job Progress", ncols=100) as pbar:
        while time.time() - start_time < timeout:
            status_data = check_job_status(job_id)

            if not status_data:
                time.sleep(check_interval)
                continue

            status = status_data.get("status")
            if status != previous_status:
                print_status(f"Job status: {status}", "info")
                previous_status = status

            # Update progress bar based on status
            if status == "completed":
                pbar.update(100 - pbar.n)  # Complete the progress
                print_status("Job completed successfully!", "success")
                return True, status_data
            elif status == "failed":
                error_msg = status_data.get("message", "Unknown error")
                print_status(f"Job failed: {error_msg}", "error")
                return False, status_data
            elif status == "processing":
                # Estimate progress
                if pbar.n < 50:
                    pbar.update(1)  # Slowly increase
                elif pbar.n < 80:
                    pbar.update(0.5)  # Even slower

            time.sleep(check_interval)

        print_status(f"Timeout after {timeout} seconds", "error")
        return False, {"status": "timeout"}

def get_queue_status():
    """Get the current queue status"""
    try:
        response = requests.get(f"{API_BASE}/queue")
        if response.status_code == 200:
            queue_data = response.json()

            print_status("Current Queue Status:", "info")
            print(f"   • Pending: {queue_data.get('pending', 0)}")
            print(f"   • Processing: {queue_data.get('processing', 0)}")
            print(f"   • Completed: {queue_data.get('completed', 0)}")
            print(f"   • Failed: {queue_data.get('failed', 0)}")
            print(f"   • Total: {queue_data.get('total', 0)}")

            return queue_data
        else:
            print_status(f"Failed to get queue status. Status: {response.status_code}", "error")
            return None
    except Exception as e:
        print_status(f"Error getting queue status: {str(e)}", "error")
        return None

def check_health():
    """Check the health status of the app"""
    try:
        response = requests.get("http://localhost:5050/health")
        if response.status_code == 200:
            health_data = response.json()

            print_status("Health Status:", "info")
            print(f"   • Status: {health_data.get('status', 'unknown')}")
            print(f"   • Device: {health_data.get('device', 'unknown')}")

            queue = health_data.get('queue', {})
            print(f"   • Queue Size: {queue.get('in_memory_size', 0)}")
            print(f"   • Is Main Process: {health_data.get('is_main_process', False)}")
            print(f"   • Is Running: {health_data.get('is_running', False)}")

            return health_data
        else:
            print_status(f"Failed to get health status. Status: {response.status_code}", "error")
            return None
    except Exception as e:
        print_status(f"Error checking health: {str(e)}", "error")
        return None

def simulate_memory_leak(size_mb=500, count=5):
    """Simulate a memory leak by creating large tensors"""
    if not torch.cuda.is_available():
        print_status("CUDA not available, cannot simulate memory leak", "error")
        return

    print_header("SIMULATING MEMORY LEAK")
    print_status(f"Creating {count} tensors of {size_mb}MB each", "warning")

    # Store tensors to prevent garbage collection
    tensors = []

    # Check memory before
    print_status("Memory before leak:", "info")
    check_gpu_memory()

    # Create large tensors
    try:
        for i in range(count):
            # Each tensor will be approximately size_mb
            num_elements = size_mb * 1024 * 1024 // 4  # 4 bytes per float32
            tensor = torch.ones(num_elements, device="cuda")
            tensors.append(tensor)

            print_status(f"Created tensor {i+1}/{count}", "info")
            check_gpu_memory()
            time.sleep(2)  # Give time for the watchdog to detect
    except Exception as e:
        print_status(f"Error creating tensors: {str(e)}", "error")

    # Check memory after
    print_status("Memory after leak:", "info")
    check_gpu_memory()

    # Return the tensors so they're not garbage collected
    return tensors

def simulate_stalled_job():
    """Simulate a stalled job by submitting a job and then causing it to stall"""
    print_header("SIMULATING STALLED JOB")

    # Submit a job
    job_id = submit_job()
    if not job_id:
        print_status("Failed to submit job for stall test", "error")
        return

    print_status("Waiting for job to start processing...", "info")
    # Wait for job to enter processing state
    processing = False
    max_wait = 60  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        status_data = check_job_status(job_id)
        if status_data and status_data.get("status") == "processing":
            processing = True
            break
        time.sleep(2)

    if not processing:
        print_status("Job didn't start processing within expected time", "error")
        return

    print_status("Job is now processing, waiting for watchdog to detect stalled job...", "warning")
    print_status("Note: This test will take at least 15 minutes to complete", "info")
    print_status("The watchdog is configured to detect stalled jobs after 15 minutes", "info")

    # Wait and periodically check job status
    print_status("Monitoring job status to see if watchdog recovers it...", "info")

    recovery_timeout = 1200  # 20 minutes
    check_interval = 30  # seconds
    start_time = time.time()

    while time.time() - start_time < recovery_timeout:
        status_data = check_job_status(job_id)
        if not status_data:
            print_status("Could not check job status", "error")
        else:
            status = status_data.get("status")
            print_status(f"Current job status: {status}", "info")

            elapsed_time = time.time() - start_time
            print_status(f"Elapsed time: {elapsed_time:.1f} seconds", "info")

            if status == "failed":
                error_msg = status_data.get("message", "Unknown error")
                if "timeout" in error_msg.lower() or "recovery" in error_msg.lower():
                    print_status("Watchdog successfully recovered the stalled job!", "success")
                    return True
                else:
                    print_status(f"Job failed for a different reason: {error_msg}", "error")
                    return False
            elif status == "completed":
                print_status("Job completed successfully before watchdog timeout", "success")
                return False  # Not recovered by watchdog

        time.sleep(check_interval)

    print_status("Test timed out waiting for job recovery", "error")
    return False

def load_test(num_jobs=10, delay=2):
    """Submit multiple jobs in quick succession to test queue management"""
    print_header(f"LOAD TEST - SUBMITTING {num_jobs} JOBS")

    job_ids = []
    for i in range(num_jobs):
        print_status(f"Submitting job {i+1}/{num_jobs}", "info")
        job_id = submit_job()
        if job_id:
            job_ids.append(job_id)
        time.sleep(delay)  # Small delay between submissions

    print_status(f"Submitted {len(job_ids)} jobs successfully", "success")

    # Monitor queue status
    print_status("Monitoring queue status...", "info")
    start_time = time.time()
    timeout = num_jobs * 120  # Approximately 2 minutes per job

    completed = 0
    failed = 0

    while time.time() - start_time < timeout:
        queue_status = get_queue_status()
        check_gpu_memory()

        if queue_status:
            completed = queue_status.get("completed", 0)
            failed = queue_status.get("failed", 0)
            pending = queue_status.get("pending", 0)
            processing = queue_status.get("processing", 0)

            # Check if all jobs are processed
            if completed + failed >= len(job_ids) and pending == 0 and processing == 0:
                print_status("All jobs have been processed!", "success")
                break

        time.sleep(10)  # Check every 10 seconds

    # Final status
    print_status("Load test results:", "info")
    print(f"   • Total jobs submitted: {len(job_ids)}")
    print(f"   • Completed successfully: {completed}")
    print(f"   • Failed: {failed}")

    return completed, failed

def main():
    """Main function to run the tests"""
    parser = argparse.ArgumentParser(description="GPU Watchdog Test Script")
    parser.add_argument("--test", choices=["memory", "stall", "load", "all"], default="all",
                      help="Type of test to run")
    parser.add_argument("--jobs", type=int, default=5, help="Number of jobs for load test")
    args = parser.parse_args()

    print_header("GPU WATCHDOG TEST SCRIPT")

    # Initial status check
    check_health()
    check_gpu_memory()
    get_queue_status()

    if args.test == "memory" or args.test == "all":
        # Test 1: Simulate memory leak
        simulate_memory_leak()
        time.sleep(10)  # Give watchdog time to react
        check_gpu_memory()  # Check if memory was recovered

    if args.test == "stall" or args.test == "all":
        # Test 2: Simulate stalled job
        simulate_stalled_job()

    if args.test == "load" or args.test == "all":
        # Test 3: Load test with multiple jobs
        load_test(args.jobs)

    print_header("TEST COMPLETED")
    check_health()
    check_gpu_memory()
    get_queue_status()

if __name__ == "__main__":
    main()