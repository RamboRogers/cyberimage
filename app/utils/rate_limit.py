"""
Rate limiting utilities for CyberImage
"""
import time
from typing import Dict, Tuple
from functools import wraps
from flask import request, jsonify, current_app
from werkzeug.exceptions import TooManyRequests

class RateLimiter:
    """Rate limiting implementation"""

    def __init__(self):
        self._requests: Dict[str, list] = {}
        self._window_size = 3600  # 1 hour window

    def _cleanup_old_requests(self, ip: str):
        """Remove requests older than the window size"""
        if ip in self._requests:
            current_time = time.time()
            self._requests[ip] = [
                req_time for req_time in self._requests[ip]
                if current_time - req_time < self._window_size
            ]

    def add_request(self, ip: str) -> Tuple[bool, int]:
        """Add a request and check if rate limit is exceeded"""
        self._cleanup_old_requests(ip)

        current_time = time.time()
        if ip not in self._requests:
            self._requests[ip] = []

        # Get rate limits from config
        hourly_limit = current_app.config.get("RATE_LIMIT_HOURLY", 10)

        # Check if limit is exceeded
        if len(self._requests[ip]) >= hourly_limit:
            # Calculate time until next available slot
            oldest_request = self._requests[ip][0]
            time_until_reset = self._window_size - (current_time - oldest_request)
            return False, int(time_until_reset)

        # Add new request
        self._requests[ip].append(current_time)
        return True, 0

# Create global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(f):
    """Decorator to apply rate limiting to routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip rate limiting if disabled in config
        if not current_app.config.get("ENABLE_RATE_LIMIT", True):
            return f(*args, **kwargs)

        ip = request.remote_addr
        allowed, wait_time = rate_limiter.add_request(ip)

        if not allowed:
            response = {
                "error": "Rate limit exceeded",
                "wait_time": wait_time,
                "message": f"Please try again in {wait_time} seconds"
            }
            return jsonify(response), 429

        return f(*args, **kwargs)

    return decorated_function