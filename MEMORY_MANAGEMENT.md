# CyberImage Memory Management

This document describes the job monitoring and recovery systems in the CyberImage application.

## Overview

To address stalled jobs and ensure reliable image generation, we've implemented several robust mechanisms:

1. **Job Watchdog**: Continuously monitors job execution, automatically recovering from stalled jobs.
2. **Enhanced Error Handling**: Improved cleanup procedures during normal and error conditions.
3. **Process Management**: Better handling of process termination to ensure all resources are released.
4. **Job Recovery**: Mechanisms to reset stalled jobs and retry failed ones.

## Important Note on GPU Memory Usage

**100% GPU memory usage is normal and expected** when running Flux models. The models are designed to utilize all available GPU memory for optimal performance. Our monitoring system focuses on detecting stalled jobs rather than memory usage thresholds.

## Job Watchdog (`app/utils/watchdog.py`)

The Job Watchdog provides automatic monitoring and recovery for stalled jobs:

- **Job Monitoring**: Tracks job execution time and detects stalled jobs.
- **Stalled Job Detection**: Identifies jobs that have been processing for too long (default: 15 minutes).
- **Emergency Recovery**: Resets stalled jobs and cleans up resources when necessary.

Configuration parameters:
- `max_stalled_time`: Maximum time a job can be processing (default: 900 seconds / 15 minutes)
- `check_interval`: Time between health checks (default: 30 seconds)
- `recovery_cooldown`: Minimum time between emergency recoveries (default: 300 seconds)

## Enhanced Error Handling (`app/models/manager.py`)

The ModelManager now includes:

- **Improved Error Handling**: Better cleanup in both normal and error paths.
- **Forced Cleanup**: Resource reclamation when needed.

Key methods:
- `_force_memory_cleanup()`: Frees GPU resources when necessary.

## Process Management (`run.py`)

The application startup has been enhanced to:

- **Set Memory Environment Variables**: Optimizes PyTorch memory allocation.
- **Register Cleanup Handlers**: Ensures resources are released on shutdown.
- **Handle Signals**: Properly manages termination signals.

## Integration with GenerationPipeline

The GenerationPipeline now:

- **Initializes the Watchdog**: Starts monitoring when the application launches.
- **Resets Stalled Jobs**: Recovers jobs that were processing during previous crashes.
- **Manages Clean Shutdown**: Ensures the watchdog is properly stopped on application exit.

## Testing

A test script is provided to verify the new features:

```bash
python test_watchdog.py [--test {stall,load,all}] [--jobs NUM_JOBS]
```

This script can:
- Simulate stalled jobs (note: this test takes at least 15 minutes to complete)
- Run load tests with multiple jobs

## Best Practices

To ensure stable operation:

1. Run the application with a single worker and single thread in production:
   ```
   gunicorn -w 1 --threads=1 -b 0.0.0.0:5050 --timeout=120 run:app
   ```

   This ensures that only one request is processed at a time, preventing memory issues from concurrent processing.

2. Monitor job status regularly:
   ```
   curl http://localhost:5050/api/queue
   ```

3. If issues persist, you can manually reset the application:
   ```
   sudo systemctl restart cyberimage
   ```

## Debugging Job Issues

If you encounter stalled jobs:

1. Check the application logs for watchdog activity.
2. Verify that the watchdog is detecting and recovering stalled jobs.
3. Consider adjusting the `max_stalled_time` parameter if your jobs legitimately need more time to complete.

## Future Improvements

Potential future enhancements:

1. Job retry with exponential backoff
2. Process-specific resource isolation
3. Job execution analytics and reporting
4. Dynamic adjustment of model settings based on job complexity