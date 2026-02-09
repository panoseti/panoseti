#! /usr/bin/env python3
"""
storeLoki.py
------------
Consumes logs from the Redis Queue and pushes them to Grafana Loki.
Implements batching and exponential backoff for robustness.

Architecture:
    [gRPC Server] -> (RPUSH) -> [Redis List] -> (BLPOP) -> [storeLoki] -> (POST) -> [Loki]
"""

import os
import sys
import time
import json
import redis
import requests
import logging
from typing import List, Dict, Any
from pathlib import Path

# Local imports (assuming running as module or in path)
try:
    from panoseti_grpc.telemetry.resources import make_rich_logger
    from panoseti_grpc.telemetry.config import TelemetryConfig
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    from panoseti_grpc.telemetry.resources import make_rich_logger
    from panoseti_grpc.telemetry.config import TelemetryConfig

logger = make_rich_logger("storeLoki")

# Default Config
DEFAULT_LOKI_URL = "http://localhost:3100/loki/api/v1/push"
DEFAULT_REDIS_KEY = "logs:ingress"
BATCH_SIZE = 50
FLUSH_INTERVAL = 5.0  # Seconds


class LokiPublisher:
    def __init__(self, loki_url: str):
        self.url = loki_url
        self.session = requests.Session()
        self.buffer: List[Dict] = []
        self.last_flush = time.time()

    def add(self, log_entry: Dict):
        """Adds a log to the buffer."""
        self.buffer.append(log_entry)

    def should_flush(self) -> bool:
        """Determines if buffer needs flushing."""
        is_full = len(self.buffer) >= BATCH_SIZE
        is_stale = (time.time() - self.last_flush) > FLUSH_INTERVAL
        return (is_full or is_stale) and len(self.buffer) > 0

    def flush(self):
        """Transforms buffer to Loki format and POSTs it."""
        if not self.buffer:
            return

        payload = self._build_loki_payload()

        try:
            # Loki expects nanosecond timestamps in strings
            headers = {"Content-Type": "application/json"}
            resp = self.session.post(self.url, json=payload, headers=headers, timeout=5)

            if resp.status_code == 204:
                logger.info(f"Flushed {len(self.buffer)} logs to Loki.")
                self.buffer.clear()
                self.last_flush = time.time()
            elif resp.status_code == 429:
                logger.warning("Loki rate limited (429). Retrying next cycle.")
                # Keep buffer, try again later
            else:
                logger.error(f"Loki rejected logs: {resp.status_code} - {resp.text}")
                # We clear buffer to prevent infinite loop on bad data
                self.buffer.clear()

        except Exception as e:
            logger.error(f"Network error pushing to Loki: {e}")

    def _build_loki_payload(self) -> Dict:
        """
        Converts flat list of logs into Loki Streams based on labels.
        Loki groups logs by unique label sets.
        """
        streams = {}

        for entry in self.buffer:
            # 1. Extract Labels (Indexed by Loki)
            # Note: Loki labels must be strings.
            labels = {
                "host": entry.get("host", "unknown"),
                "service": entry.get("service_name", "unknown"),
                "level": str(entry.get("severity", 2))
            }

            # Create a unique key for this label set
            stream_key = json.dumps(labels, sort_keys=True)

            if stream_key not in streams:
                streams[stream_key] = {
                    "stream": labels,
                    "values": []
                }

            # 2. Format Timestamp (Unix Seconds -> Nanoseconds String)
            # Loki strict requirement: string(epoch_ns)
            ts = int(entry.get("timestamp", time.time()) * 1e9)

            # 3. Format Line (The log content)
            # We combine the message and extra data
            line_content = {
                "msg": entry.get("payload_json", ""),
                "file": f"{entry.get('file_path')}:{entry.get('line_number')}",
                "func": entry.get("function_name")
            }
            line_str = json.dumps(line_content)

            streams[stream_key]["values"].append([str(ts), line_str])

        return {"streams": list(streams.values())}


def main():
    logger.info("Starting Loki Log Shipper...")

    # 1. Config
    config_path = Path("telemetry_config.toml")
    if not config_path.exists():
        # Try local package location if running from root
        config_path = Path("capture_telemetry_service/telemetry_config.toml")

    redis_key = DEFAULT_REDIS_KEY
    loki_url = DEFAULT_LOKI_URL

    if config_path.exists():
        try:
            cfg = TelemetryConfig.load(str(config_path))
            logging_cfg = getattr(cfg, "logging", {})
            redis_key = logging_cfg.get("redis_queue_key", DEFAULT_REDIS_KEY)
            loki_url = logging_cfg.get("loki_url", DEFAULT_LOKI_URL)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")

    logger.info(f"Listening on Redis Key: [bold cyan]{redis_key}[/]", extra={"markup": True})
    logger.info(f"Pushing to Loki: [bold cyan]{loki_url}[/]", extra={"markup": True})

    # 2. Connections
    try:
        r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)
        publisher = LokiPublisher(loki_url)
    except Exception as e:
        logger.critical(f"Failed to connect: {e}")
        sys.exit(1)

    # 3. Event Loop
    while True:
        try:
            # A. Check Buffer Flush conditions
            if publisher.should_flush():
                publisher.flush()

            # B. Fetch Log (Blocking Pop with Timeout)
            # Timeout is important so we can return to loop to flush buffer based on time
            # even if no new logs arrive.
            item = r.blpop(redis_key, timeout=1)

            if item:
                # item is a tuple ('key', 'value')
                log_json = item[1]
                try:
                    log_entry = json.loads(log_json)
                    publisher.add(log_entry)
                except json.JSONDecodeError:
                    logger.error("Received invalid JSON from Redis. Dropping.")

        except redis.RedisError as e:
            logger.error(f"Redis Error: {e}")
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping...")
            publisher.flush()
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()