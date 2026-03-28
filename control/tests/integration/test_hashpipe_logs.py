"""
test_hashpipe_logs.py — Integration tests for hashpipe log forwarding.

The daq_control gRPC server uses get_logger("daq_control", grpc_enabled=True),
which forwards log records via the Telemetry gRPC service to the headnode's
capture_telemetry_service daemon. That daemon pushes logs to:
  1. Redis (logs:ingress list) — immediate
  2. Loki  (via storeLoki.py) — within ~10s

These tests verify that a StartDaq/StopDaq cycle produces log entries
visible on the headnode.
"""
from __future__ import annotations

import time

import pytest
import requests

from .conftest import LOKI_URL, REDIS_HOST


def _loki_query(query: str, limit: int = 50) -> list:
    """Return Loki log stream results for a LogQL query."""
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


def _wait_for_loki(query: str, timeout: float = 20.0) -> bool:
    """Poll Loki until results appear or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _loki_query(query):
            return True
        time.sleep(1)
    return False


class TestHashpipeLogs:

    def test_startdaq_log_arrives_in_loki(self, daq_control_direct, run_params):
        """StartDaq/StopDaq generates log entries visible in Loki within 20s."""
        daq_control_direct.StartDaq(run_params)
        time.sleep(1)
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        time.sleep(2)  # let gRPC telemetry flush

        found = _wait_for_loki('{service="daq_control"}', timeout=20)
        assert found, "daq_control logs did not appear in Loki within 20s"

    def test_log_entry_contains_run_dir(self, daq_control_direct, run_params):
        """A log entry for StartDaq includes the run_dir string in its payload."""
        daq_control_direct.StartDaq(run_params)
        run_dir = run_params["run_dir"]
        time.sleep(1)
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_dir,
        })
        time.sleep(2)

        deadline = time.time() + 20
        while time.time() < deadline:
            results = _loki_query('{service="daq_control"}', limit=100)
            for stream in results:
                for _, line in stream.get("values", []):
                    if run_dir in line:
                        return
            time.sleep(1)
        pytest.fail(
            f"No log entry mentioning run_dir={run_dir!r} found in Loki within 20s"
        )

    def test_logs_appear_in_redis_before_loki(self, daq_control_direct, run_params):
        """
        After StartDaq/StopDaq, the Redis logs:ingress list grows,
        confirming that gRPC log forwarding reaches Redis before storeLoki ships to Loki.
        """
        try:
            import redis as redis_lib
        except ImportError:
            pytest.skip("redis package not installed")

        r = redis_lib.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
        initial_len = r.llen("logs:ingress")

        daq_control_direct.StartDaq(run_params)
        time.sleep(2)
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        time.sleep(3)  # let telemetry flush to Redis

        new_len = r.llen("logs:ingress")
        assert new_len > initial_len, (
            f"Expected new log entries in Redis logs:ingress "
            f"(was {initial_len}, now {new_len})"
        )
