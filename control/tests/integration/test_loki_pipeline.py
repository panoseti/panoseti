"""
test_loki_pipeline.py — Integration tests for the storeLoki.py Redis→Loki pipeline.

storeLoki.py sits between Redis (logs:ingress) and Loki:
    [gRPC Telemetry] → (RPUSH) → [Redis: logs:ingress] → [storeLoki] → [Loki]

These tests inject a log entry directly into Redis and verify that storeLoki
ships it to Loki within the expected time window (≤ FLUSH_INTERVAL + network).
"""
from __future__ import annotations

import json
import time

import pytest
import requests

from .conftest import LOKI_URL, REDIS_HOST


def _loki_query(query: str, limit: int = 50) -> list:
    try:
        resp = requests.get(
            f"{LOKI_URL}/loki/api/v1/query_range",
            params={"query": query, "limit": limit},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("data", {}).get("result", [])
    except requests.RequestException:
        pass
    return []


@pytest.fixture(scope="module")
def redis_client():
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
        r.ping()
        return r
    except Exception as e:
        pytest.skip(f"Redis unavailable: {e}")


class TestLokiPipeline:

    def test_log_entry_ships_to_loki(self, redis_client):
        """storeLoki ships a Redis log entry to Loki within 15s."""
        service = f"ci_pipeline_test_{int(time.time())}"
        entry = {
            "service_name":  service,
            "payload_json":  '{"msg": "integration test"}',
            "timestamp":     time.time(),
            "host":          "test-runner",
            "severity":      2,
            "function_name": "test_log_entry_ships_to_loki",
            "git_branch":    "ci",
            "git_commit":    "unknown",
        }
        redis_client.rpush("logs:ingress", json.dumps(entry))

        deadline = time.time() + 15
        while time.time() < deadline:
            results = _loki_query(f'{{service="{service}"}}')
            if results:
                return
            time.sleep(1)
        pytest.fail(f"Log with service={service!r} did not appear in Loki within 15s")

    def test_multiple_entries_all_arrive(self, redis_client):
        """All entries from a batch arrive in Loki (batch ≤ BATCH_SIZE=100)."""
        service = f"ci_batch_test_{int(time.time())}"
        n_entries = 5
        for i in range(n_entries):
            entry = {
                "service_name":  service,
                "payload_json":  json.dumps({"seq": i}),
                "timestamp":     time.time(),
                "host":          "test-runner",
                "severity":      2,
                "function_name": "test_multiple_entries_all_arrive",
                "git_branch":    "ci",
                "git_commit":    "unknown",
            }
            redis_client.rpush("logs:ingress", json.dumps(entry))

        # Wait for all to arrive
        deadline = time.time() + 15
        while time.time() < deadline:
            results = _loki_query(f'{{service="{service}"}}', limit=n_entries + 5)
            total = sum(len(stream.get("values", [])) for stream in results)
            if total >= n_entries:
                return
            time.sleep(1)
        pytest.fail(
            f"Expected {n_entries} entries for service={service!r} in Loki, "
            f"but pipeline did not deliver them within 15s"
        )

    def test_invalid_json_does_not_crash_pipeline(self, redis_client):
        """Pushing invalid JSON to Redis should not crash storeLoki."""
        service = f"ci_after_invalid_{int(time.time())}"

        # Push garbage
        redis_client.rpush("logs:ingress", "this is not json {{{")
        time.sleep(3)  # let storeLoki process the bad entry

        # Push a valid entry — storeLoki should still be running
        entry = {
            "service_name":  service,
            "payload_json":  '{"msg": "post-invalid check"}',
            "timestamp":     time.time(),
            "host":          "test-runner",
            "severity":      2,
            "function_name": "test_invalid_json_does_not_crash_pipeline",
            "git_branch":    "ci",
            "git_commit":    "unknown",
        }
        redis_client.rpush("logs:ingress", json.dumps(entry))

        deadline = time.time() + 15
        while time.time() < deadline:
            if _loki_query(f'{{service="{service}"}}'):
                return
            time.sleep(1)
        pytest.fail("storeLoki did not recover after invalid JSON — pipeline may have crashed")
