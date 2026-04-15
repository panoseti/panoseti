"""
test_hashpipe_logs.py — Integration tests for hashpipe log forwarding.

The daqnode runs the unified panoseti-server (daq_data + daq_control) with
grpc_logging=true.  Log records are forwarded via gRPC to the headnode's
Telemetry service (panoseti-server --profile headnode at 10.0.1.22:50051),
which writes them to Redis (logs:ingress). storeLoki.py then ships them to Loki.

"""
from __future__ import annotations

import pytest
import requests

from .conftest import (
    LOKI_URL,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
    wait_until,
)


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


def _wait_for_loki(query: str, timeout: float = 30.0) -> bool:
    """Poll Loki until results appear or timeout."""
    return wait_until(lambda: bool(_loki_query(query)), timeout=timeout, interval=0.5)


class TestHashpipeLogs:

    def test_startdaq_log_arrives_in_loki(self, daq_control_direct, run_params, ensure_clean_daq_state):
        """StartDaq/StopDaq generates log entries visible in Loki within 30s."""
        daq_control_direct.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_direct, run_params["data_dir"]), (
            "hashpipe did not start"
        )
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        # Wait for hashpipe to stop (so daq_control flushes its final log records)
        wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"])

        found = _wait_for_loki('{service="daq_control_server"}', timeout=30)
        assert found, "daq_control logs did not appear in Loki within 30s"

    def test_log_entry_contains_run_dir(self, daq_control_direct, run_params, ensure_clean_daq_state):
        """A log entry for StartDaq includes the run_dir string in its payload."""
        daq_control_direct.StartDaq(run_params)
        run_dir = run_params["run_dir"]
        assert wait_hashpipe_running(daq_control_direct, run_params["data_dir"]), (
            "hashpipe did not start"
        )
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_dir,
        })
        wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"])

        found = wait_until(
            lambda: any(
                run_dir in line
                for stream in _loki_query('{service="daq_control_server"}', limit=100)
                for _, line in stream.get("values", [])
            ),
            timeout=30,
            interval=0.5,
        )
        if not found:
            pytest.fail(
                f"No log entry mentioning run_dir={run_dir!r} found in Loki within 30s"
            )

    

    # def test_logs_appear_in_redis_before_loki(self, daq_control_direct, run_params):
    #     """
    #     After StartDaq/StopDaq, the Redis logs:ingress list grows,
    #     confirming that gRPC log forwarding reaches Redis before storeLoki ships to Loki.
    #     """
    #     try:
    #         import redis as redis_lib
    #     except ImportError:
    #         pytest.skip("redis package not installed")

    #     r = redis_lib.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    #     initial_len = r.llen("logs:ingress")

    #     daq_control_direct.StartDaq(run_params)
    #     assert wait_hashpipe_running(daq_control_direct, run_params["data_dir"]), (
    #         "hashpipe did not start"
    #     )
    #     daq_control_direct.StopDaq({
    #         "data_dir": run_params["data_dir"],
    #         "run_dir":  run_params["run_dir"],
    #     })
    #     # Poll Redis directly: far more reliable than a fixed sleep
    #     grew = wait_until(
    #         lambda: r.llen("logs:ingress") > initial_len,
    #         timeout=15,
    #         interval=0.25,
    #     )
    #     new_len = r.llen("logs:ingress")
    #     assert grew, (
    #         f"Expected new log entries in Redis logs:ingress "
    #         f"(was {initial_len}, now {new_len}) — "
    #         "check HEADNODE_IP reachability from daqnode and Telemetry gRPC health"
    #     )
