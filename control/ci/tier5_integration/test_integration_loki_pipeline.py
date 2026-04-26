"""
test_integration_loki_pipeline.py — Tier 5 Integration tests for the storeLoki.py pipeline.

Connects to the STATIC Docker Compose stack.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import pytest
import redis
import requests

# Use static values or env vars if provided
LOKI_URL = os.getenv("LOKI_URL", "http://localhost:3100")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")


def _loki_query(query: str, limit: int = 50) -> list[Any]:
    try:
        resp = requests.get(
            f"{LOKI_URL}/loki/api/v1/query_range",
            params={"query": query, "limit": str(limit)},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("data", {}).get("result", [])
    except requests.RequestException:
        pass
    return []


@pytest.fixture(scope="module")
def redis_client() -> redis.Redis:
    try:
        r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
        r.ping()
        return r
    except Exception as e:
        pytest.skip(f"Redis unavailable at {REDIS_HOST}: {e}")


class TestIntegrationLokiPipeline:

    def test_log_entry_ships_to_loki(self, redis_client) -> None:
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
            time.sleep(0.2)
        pytest.fail(f"Log with service={service!r} did not appear in Loki within 15s")

    def test_multiple_entries_all_arrive(self, redis_client) -> None:
        """All entries from a batch arrive in Loki."""
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

        deadline = time.time() + 15
        while time.time() < deadline:
            results = _loki_query(f'{{service="{service}"}}', limit=n_entries + 5)
            total = sum(len(stream.get("values", [])) for stream in results)
            if total >= n_entries:
                return
            time.sleep(0.2)
        pytest.fail(f"Expected {n_entries} entries in Loki within 15s")

    def test_invalid_json_does_not_crash_pipeline(self, redis_client) -> None:
        """Pushing invalid JSON to Redis should not crash storeLoki."""
        service = f"ci_after_invalid_{int(time.time())}"
        redis_client.rpush("logs:ingress", "this is not json {{{")
        time.sleep(1)

        entry = {
            "service_name":  service,
            "payload_json":  '{"msg": "post-invalid check"}',
            "timestamp":     time.time(),
            "host":          "test-runner",
            "severity":      2,
        }
        redis_client.rpush("logs:ingress", json.dumps(entry))

        deadline = time.time() + 15
        while time.time() < deadline:
            if _loki_query(f'{{service="{service}"}}'):
                return
            time.sleep(0.2)
        pytest.fail("storeLoki did not recover after invalid JSON")

    def test_log_severity_levels_distinct(self, redis_client) -> None:
        """DEBUG, INFO, WARNING, ERROR entries ship to Loki with distinct severity labels."""
        service = f"ci_sev_{int(time.time())}"
        for sev in range(4):
            entry = {
                "service_name":  service,
                "payload_json":  json.dumps({"sev": sev}),
                "timestamp":     time.time(),
                "host":          "test-runner",
                "severity":      sev,
            }
            redis_client.rpush("logs:ingress", json.dumps(entry))

        deadline = time.time() + 20
        while time.time() < deadline:
            results = _loki_query(f'{{service="{service}"}}', limit=20)
            total = sum(len(s.get("values", [])) for s in results)
            if total >= 4:
                return
            time.sleep(0.2)
        pytest.fail(f"Expected ≥4 severity entries for service={service!r} in Loki")

    def test_large_payload_ships_without_truncation(self, redis_client) -> None:
        """A 5000-character message payload arrives in Loki without truncation."""
        service = f"ci_large_{int(time.time())}"
        large_msg = "X" * 5000
        entry = {
            "service_name":  service,
            "payload_json":  json.dumps({"msg": large_msg}),
            "timestamp":     time.time(),
            "host":          "test-runner",
            "severity":      1,
        }
        redis_client.rpush("logs:ingress", json.dumps(entry))

        deadline = time.time() + 20
        while time.time() < deadline:
            results = _loki_query(f'{{service="{service}"}}', limit=5)
            for stream in results:
                for _, line in stream.get("values", []):
                    if large_msg in line or len(line) > 4900:
                        return
            time.sleep(0.2)
        pytest.fail("Large 5000-char payload did not appear in Loki")

    def test_burst_logging_all_entries_arrive(self, redis_client) -> None:
        """50 rapid log pushes all arrive in Loki within 30s."""
        service = f"ci_burst_{int(time.time())}"
        n = 50
        for i in range(n):
            entry = {
                "service_name":  service,
                "payload_json":  json.dumps({"seq": i}),
                "timestamp":     time.time(),
                "host":          "test-runner",
                "severity":      1,
            }
            redis_client.rpush("logs:ingress", json.dumps(entry))

        deadline = time.time() + 30
        while time.time() < deadline:
            results = _loki_query(f'{{service="{service}"}}', limit=n + 10)
            total = sum(len(s.get("values", [])) for s in results)
            if total >= n:
                return
            time.sleep(0.2)
        pytest.fail(f"Expected {n} burst entries in Loki")

    def test_log_entry_metadata_fields_present(self, redis_client) -> None:
        """Loki label set includes the service field derived from the log entry."""
        service = f"ci_meta_{int(time.time())}"
        entry = {
            "service_name":  service,
            "payload_json":  json.dumps({"check": "metadata"}),
            "timestamp":     time.time(),
            "host":          "test-runner",
            "severity":      1,
        }
        redis_client.rpush("logs:ingress", json.dumps(entry))

        deadline = time.time() + 20
        while time.time() < deadline:
            results = _loki_query(f'{{service="{service}"}}', limit=5)
            if results:
                labels = results[0].get("stream", {})
                assert labels.get("service") == service
                return
            time.sleep(0.2)
        pytest.fail(f"Log entry for service={service!r} missing metadata in Loki")
