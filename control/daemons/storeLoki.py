#!/usr/bin/env python3
"""
storeLoki.py
------------
Consumes logs from the Redis Queue and pushes them to Grafana Loki.
Implements batching, GZIP compression, and exponential backoff for robustness.

Architecture:
    [gRPC Server] -> (RPUSH) -> [Redis List] -> (LMOVE) -> [Processing List] -> [storeLoki] -> (POST) -> [Loki]
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

import redis
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from panoseti_grpc.telemetry.config import TelemetryConfig
    from panoseti_grpc.telemetry.logger import get_logger
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from panoseti_grpc.telemetry.config import TelemetryConfig
    from panoseti_grpc.telemetry.logger import get_logger

logger = get_logger("storeLoki", grpc_enabled=False)

# --- Configuration Constants ---
DEFAULT_LOKI_URL = "http://localhost:3100/loki/api/v1/push"
DEFAULT_REDIS_KEY = "logs:ingress"
PROCESSING_REDIS_KEY = "logs:processing"
BATCH_SIZE = 100          # Flush when we have 100 logs
MAX_BUFFER_SIZE = 10000   # Safety valve: clear buffer if it reaches this size
FLUSH_INTERVAL = 2.0      # Flush at least every 2 seconds
MAX_BACKOFF_SECONDS = 60  # Cap retry wait time
MAX_FLUSH_SIZE_BYTES = 512 * 1024  # 512 KB compressed limit


class LokiPublisher:
    """Handles buffering and pushing logs to Loki with GZIP compression and backoff."""

    def __init__(self, loki_url: str, redis_client: redis.Redis):
        self.url = loki_url
        self.redis = redis_client
        self.buffer: list[dict] = []
        self.last_flush = time.time()
        self.current_batch_bytes = 0

        # Backoff state
        self.consecutive_errors = 0
        self.next_retry_time = 0.0

        # Robust HTTP session with retry
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def can_accept_more(self) -> bool:
        """Returns False if buffer is full (backpressure to Redis)."""
        return len(self.buffer) < MAX_BUFFER_SIZE

    def add(self, log_entry: dict) -> None:
        self.buffer.append(log_entry)
        # Conservative estimate of uncompressed size
        self.current_batch_bytes += len(json.dumps(log_entry))

    def should_flush(self) -> bool:
        if not self.buffer:
            return False
        if time.time() < self.next_retry_time:
            return False
        
        # Flush if we hit record count, time interval, or raw size threshold
        is_full = len(self.buffer) >= BATCH_SIZE
        is_stale = (time.time() - self.last_flush) > FLUSH_INTERVAL
        is_oversized = self.current_batch_bytes >= MAX_FLUSH_SIZE_BYTES
        return is_full or is_stale or is_oversized

    def flush(self) -> None:
        """Compress buffer and POST to Loki; preserve buffer on retriable errors."""
        if not self.buffer:
            return

        payload = self._build_loki_payload()

        try:
            raw_data = json.dumps(payload).encode('utf-8')
            compressed_data = gzip.compress(raw_data)
        except (UnicodeError, ValueError, OSError) as e:
            logger.error(f"Encoding or compression failed: {e}. One or more log entries may contain invalid characters. Dropping batch.")
            self._clear_batch()
            return

        # If even a single compressed message exceeds our limit, we must drop it
        # or it will poison the queue forever.
        if len(compressed_data) > MAX_FLUSH_SIZE_BYTES and len(self.buffer) == 1:
            logger.error(f"Single log entry too large ({len(compressed_data)} bytes). Dropping.")
            self._clear_batch()
            return

        try:
            headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
            resp = self.session.post(self.url, data=compressed_data, headers=headers, timeout=10)

            if resp.status_code == 204:
                if self.consecutive_errors > 0:
                    logger.info("Reconnected to Loki.")
                self._clear_batch()
                self.last_flush = time.time()
                self.consecutive_errors = 0
                self.next_retry_time = 0.0

            elif resp.status_code == 429:
                logger.warning("Loki rate limit (429). Slowing down.")
                self._apply_backoff(initial=2.0)

            elif 400 <= resp.status_code < 500:
                logger.error(f"Loki rejected data ({resp.status_code}): {resp.text}")
                # Irrecoverable client error (e.g. malformed or oversized). Drop to unblock.
                self._clear_batch()

            else:
                logger.warning(f"Loki server error ({resp.status_code}). Retrying.")
                self._apply_backoff()

        except requests.exceptions.RequestException as e:
            if self.consecutive_errors == 0:
                logger.error(f"Loki unreachable: {e}")
            self._apply_backoff()
            # Safety Valve: Clear buffer if it gets too large to prevent OOM (SC-056)
            if len(self.buffer) >= MAX_BUFFER_SIZE:
                 logger.critical(f"Loki still unreachable and buffer full ({len(self.buffer)}). Dropping batch to prevent OOM.")
                 self._clear_batch()

    def _clear_batch(self) -> None:
        self.buffer.clear()
        self.current_batch_bytes = 0
        self.redis.delete(PROCESSING_REDIS_KEY)

    def _apply_backoff(self, initial: float = 1.0) -> None:
        self.consecutive_errors += 1
        delay = min(initial * (2 ** (self.consecutive_errors - 1)), MAX_BACKOFF_SECONDS)
        self.next_retry_time = time.time() + delay

    def _build_loki_payload(self) -> dict:
        """Group logs by labels for Loki efficiency."""
        streams: dict[str, dict] = {}

        # Sort buffer by timestamp to ensure monotonic increasing order within each stream
        self.buffer.sort(key=lambda x: x.get("timestamp", 0))

        for entry in self.buffer:
            labels = {
                "job":           "panoseti",
                "host":          entry.get("host", "unknown"),
                "service":       entry.get("service_name", "unknown"),
                "severity":      str(entry.get("severity", 2)),
                "function_name": entry.get("function_name", "unknown"),
                "git_branch":    entry.get("git_branch", "unknown"),
                "git_commit":    entry.get("git_commit", "unknown"),
            }
            stream_key = json.dumps(labels, sort_keys=True)
            if stream_key not in streams:
                streams[stream_key] = {"stream": labels, "values": []}

            try:
                ts_ns = int(entry.get("timestamp", time.time()) * 1e9)
                ts_str = str(ts_ns)
            except (ValueError, TypeError):
                ts_str = str(int(time.time() * 1e9))

            line_str = entry.get("payload_json", "{}")
            streams[stream_key]["values"].append([ts_str, line_str])

        return {"streams": list(streams.values())}


def recover_processing_queue(r: redis.Redis, ingress_key: str, processing_key: str) -> None:
    """Move all items from processing queue back to ingress queue on startup."""
    count = 0
    while True:
        # Move from processing (left) back to ingress (right)
        item = r.lmove(processing_key, ingress_key, src="LEFT", dest="RIGHT")
        if item:
            count += 1
        else:
            break
    if count > 0:
        logger.info(f"Recovered {count} logs from processing queue.")


def main() -> None:
    logger.info("Starting Loki log shipper...")

    config_path = Path("capture_telemetry_service/telemetry_config.toml")
    if not config_path.exists():
        config_path = Path("telemetry_config.toml")

    redis_key = DEFAULT_REDIS_KEY
    loki_base = os.getenv("LOKI_URL", "http://localhost:3100")
    loki_url = loki_base.rstrip("/") + "/loki/api/v1/push"

    if config_path.exists():
        try:
            TelemetryConfig.load(str(config_path))
        except (OSError, ValueError, KeyError) as e:
            logger.warning(f"Config load warning: {e} — using defaults")

    logger.info(f"Source Redis key: {redis_key}")
    logger.info(f"Target Loki URL: {loki_url}")

    try:
        r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=False)
        r.ping()
        
        # Reliable Queue Recovery
        recover_processing_queue(r, redis_key, PROCESSING_REDIS_KEY)
        
        publisher = LokiPublisher(loki_url, r)
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logger.critical(f"Redis startup failed: {e}")
        sys.exit(1)

    while True:
        try:
            if publisher.should_flush():
                publisher.flush()

            if publisher.can_accept_more():
                # Atomic reliable pop: ingress -> processing
                item = r.blmove(redis_key, PROCESSING_REDIS_KEY, timeout=1, src="RIGHT", dest="LEFT")
                if isinstance(item, (str, bytes)):
                    try:
                        # Ensure we decode bytes with replacement for invalid characters
                        if isinstance(item, bytes):
                            item_str = item.decode('utf-8', errors='replace')
                        else:
                            item_str = item
                        log_entry = json.loads(item_str)
                        publisher.add(log_entry)
                    except json.JSONDecodeError:
                        logger.error("Skipping invalid JSON log entry")
                        r.lpop(PROCESSING_REDIS_KEY) # Remove invalid item
            else:
                time.sleep(0.5)

        except redis.RedisError as e:
            logger.error(f"Redis connection error: {e}")
            time.sleep(2)
        except KeyboardInterrupt:
            logger.info("Shutdown signal. Flushing final logs...")
            publisher.flush()
            break
        except Exception as e:
            logger.exception(f"Unexpected Critical Error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
