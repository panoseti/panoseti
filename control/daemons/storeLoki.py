#! /usr/bin/env python3
"""
storeLoki.py
------------
Consumes logs from the Redis Queue and pushes them to Grafana Loki.
Implements batching, GZIP compression, and exponential backoff for robustness.

Architecture:
    [gRPC Server] -> (RPUSH) -> [Redis List] -> (BLPOP) -> [storeLoki] -> (POST) -> [Loki]
"""

import os
import sys
import time
import json
import gzip
import redis
import requests
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Optional, Any
from pathlib import Path

# Local imports
try:
    from panoseti_grpc.telemetry.resources import make_rich_logger
    from panoseti_grpc.telemetry.config import TelemetryConfig
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from panoseti_grpc.telemetry.resources import make_rich_logger
    from panoseti_grpc.telemetry.config import TelemetryConfig

logger = make_rich_logger("storeLoki")

# --- Configuration Constants ---
DEFAULT_LOKI_URL = "http://localhost:3100/loki/api/v1/push"
DEFAULT_REDIS_KEY = "logs:ingress"
BATCH_SIZE = 100  # Flush when we have 100 logs
MAX_BUFFER_SIZE = 2000  # Stop pulling from Redis if we hold this many
FLUSH_INTERVAL = 2.0  # Flush at least every 2 seconds (High responsiveness)
MAX_BACKOFF_SECONDS = 60  # Cap retry wait time


class LokiPublisher:
    """
    Handles buffering and pushing logs to Loki with GZIP compression and backoff.
    """

    def __init__(self, loki_url: str):
        self.url = loki_url
        self.buffer: List[Dict] = []
        self.last_flush = time.time()

        # Backoff State
        self.consecutive_errors = 0
        self.next_retry_time = 0

        # Setup robust HTTP session
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def can_accept_more(self) -> bool:
        """Returns False if buffer is full (Backpressure to Redis)."""
        return len(self.buffer) < MAX_BUFFER_SIZE

    def add(self, log_entry: Dict) -> None:
        """Adds a log to the buffer."""
        self.buffer.append(log_entry)

    def should_flush(self) -> bool:
        """Determines if buffer needs flushing based on size or time."""
        if not self.buffer:
            return False

        # 1. Backoff Check: If we failed recently, don't try yet
        if time.time() < self.next_retry_time:
            return False

        # 2. Size/Time Check
        is_full = len(self.buffer) >= BATCH_SIZE
        is_stale = (time.time() - self.last_flush) > FLUSH_INTERVAL
        return is_full or is_stale

    def flush(self) -> None:
        """
        Compresses buffer and POSTs to Loki.
        Handles errors by preserving the buffer for the next retry (unless fatal).
        """
        if not self.buffer:
            return

        payload = self._build_loki_payload()

        # Compress payload
        try:
            compressed_data = gzip.compress(json.dumps(payload).encode('utf-8'))
        except Exception as e:
            logger.error(f"Compression failed: {e}. Dropping batch.")
            self.buffer.clear()
            return

        try:
            headers = {
                "Content-Type": "application/json",
                "Content-Encoding": "gzip"
            }

            # Send Request
            resp = self.session.post(
                self.url,
                data=compressed_data,
                headers=headers,
                timeout=5
            )

            if resp.status_code == 204:
                # SUCCESS
                if self.consecutive_errors > 0:
                    logger.info("Reconnected to Loki.")

                self.buffer.clear()
                self.last_flush = time.time()
                self.consecutive_errors = 0
                self.next_retry_time = 0

            elif resp.status_code == 429:
                # Rate Limit: Wait a bit, but don't drop data
                logger.warning("Loki Rate Limit (429). slowing down.")
                self._apply_backoff(initial=2.0)

            elif 400 <= resp.status_code < 500:
                # FATAL Client Error (Bad Request): Data is rejected.
                # Retrying won't fix this. Drop buffer to prevent loop.
                logger.error(f"Loki Rejected Data ({resp.status_code}): {resp.text}")
                self.buffer.clear()

            else:
                # Server Error (5xx): Retry
                logger.warning(f"Loki Server Error ({resp.status_code}). Retrying.")
                self._apply_backoff()

        except requests.exceptions.RequestException as e:
            # Network Error (Connection Refused, Timeout): Retry
            # Only log error if it's new, to avoid spamming logs
            if self.consecutive_errors == 0:
                logger.error(f"Loki Unreachable: {e}")
            self._apply_backoff()

    def _apply_backoff(self, initial: float = 1.0) -> None:
        """Calculates exponential sleep time."""
        self.consecutive_errors += 1
        # Exponential: 1, 2, 4, 8, ... up to 60s
        delay = min(initial * (2 ** (self.consecutive_errors - 1)), MAX_BACKOFF_SECONDS)
        self.next_retry_time = time.time() + delay

    def _build_loki_payload(self) -> Dict:
        """Groups logs by labels for Loki efficiency."""
        streams = {}

        for entry in self.buffer:
            labels = {
                "host": entry.get("host", "unknown"),
                "service": entry.get("service_name", "unknown"),
                "severity": str(entry.get("severity", 2)),
                # Optional: Add trace_id if available
                "git_commit": entry.get("git_commit", "unknown")
            }

            stream_key = json.dumps(labels, sort_keys=True)
            if stream_key not in streams:
                streams[stream_key] = {"stream": labels, "values": []}

            # Valid Nanosecond Timestamp
            try:
                ts_ns = int(entry.get("timestamp", time.time()) * 1e9)
                ts_str = str(ts_ns)
            except (ValueError, TypeError):
                ts_str = str(int(time.time() * 1e9))

            # Use the entire original JSON as the log line content
            line_str = entry.get("payload_json", "{}")

            streams[stream_key]["values"].append([ts_str, line_str])

        return {"streams": list(streams.values())}


def main():
    logger.info("Starting Optimized Loki Log Shipper...")

    # 1. Config Loading
    config_path = Path("telemetry_config.toml")
    if not config_path.exists():
        # Fallback path for monorepo structure
        fallback = Path("capture_telemetry_service/telemetry_config.toml")
        if fallback.exists():
            config_path = fallback

    redis_key = DEFAULT_REDIS_KEY
    loki_url = DEFAULT_LOKI_URL

    if config_path.exists():
        try:
            cfg = TelemetryConfig.load(str(config_path))
            # Assuming config object structure; adjust if needed based on actual TelemetryConfig
            # For now, we manually parse if the object doesn't expose it directly
            pass
        except Exception as e:
            logger.warning(f"Config load warning: {e}")

    logger.info(f"Source Redis: [bold cyan]{redis_key}[/]", extra={"markup": True})
    logger.info(f"Target Loki:  [bold cyan]{loki_url}[/]", extra={"markup": True})

    # 2. Connections
    try:
        r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)
        # Test connection
        r.ping()
        publisher = LokiPublisher(loki_url)
    except Exception as e:
        logger.critical(f"Startup Failed: {e}")
        sys.exit(1)

    # 3. Main Loop
    while True:
        try:
            # A. Flush if time/size limit reached
            if publisher.should_flush():
                publisher.flush()

            # B. Pull from Redis ONLY if buffer has space
            # This provides backpressure: if Loki is down, we stop draining Redis.
            if publisher.can_accept_more():
                # Blocking pop with short timeout (1s) allows checking flush timer frequently
                item = r.blpop(redis_key, timeout=1)

                if item:
                    # item is ('logs:ingress', 'json_string')
                    try:
                        log_entry = json.loads(item[1])
                        publisher.add(log_entry)
                    except json.JSONDecodeError:
                        logger.error("Skipping invalid JSON log entry")
            else:
                # Buffer is full (likely Loki down). Sleep briefly to avoid busy loop.
                time.sleep(0.5)

        except redis.RedisError as e:
            logger.error(f"Redis Connection Error: {e}")
            time.sleep(2)
        except KeyboardInterrupt:
            logger.info("Shutdown Signal Received. Flushing final logs...")
            publisher.flush()
            break
        except Exception as e:
            logger.exception(f"Unexpected Critical Error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()