"""
test_gateway_topology.py — Integration tests for the socat gateway (NAT/VPN) topology.

Extends beyond the parameterized TestDaqLifecycle to cover gateway-specific scenarios:
- Gateway client can reach the daqnode (TCP forwarding works end-to-end)
- Direct and gateway clients observe the same server state
- StopDaq via gateway shuts down hashpipe visible from the direct client

Topology (from docker-compose.integration.yml):
    test-runner (10.0.1.5) → gateway (10.0.1.254) → daqnode (192.168.0.10)
    test-runner (192.168.0.5) ─────────────────────→ daqnode (192.168.0.10)
"""
from __future__ import annotations

import time

import pytest

from panoseti_grpc.daq_control.client import DaqControlClient

from .conftest import (
    DAQNODE_DIRECT_HOST, DAQNODE_GATEWAY_HOST, GRPC_PORT,
    wait_hashpipe_running, wait_hashpipe_stopped,
)


class TestGatewayForwarding:
    """Gateway (socat) client reaches the daqnode and observes consistent state."""

    def test_gateway_client_starts_daq(self, daq_control_gateway, run_params, ensure_clean_daq_state):
        """DaqControlClient via socat gateway can issue StartDaq successfully."""
        ok = daq_control_gateway.StartDaq(run_params)
        assert ok is True

    def test_gateway_client_reports_running(self, daq_control_gateway, run_params, ensure_clean_daq_state):
        """After StartDaq via gateway, StatusDaq also via gateway sees hashpipe_running=True."""
        daq_control_gateway.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_gateway, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        ok, status = daq_control_gateway.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        assert ok
        assert status.get("hashpipe_running") is True

    def test_direct_and_gateway_see_same_state(
        self, daq_control_direct, daq_control_gateway, run_params, ensure_clean_daq_state
    ):
        """Direct and gateway clients report the same hashpipe_running state."""
        daq_control_direct.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_direct, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )

        status_request = {
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        }
        _, s_direct  = daq_control_direct.StatusDaq(status_request)
        _, s_gateway = daq_control_gateway.StatusDaq(status_request)

        assert s_direct.get("hashpipe_running") is True
        assert s_gateway.get("hashpipe_running") is True
        assert s_direct["hashpipe_running"] == s_gateway["hashpipe_running"]

    def test_gateway_stop_is_visible_to_direct(
        self, daq_control_direct, daq_control_gateway, run_params, ensure_clean_daq_state
    ):
        """StopDaq issued via gateway makes hashpipe_running=False on the direct client."""
        daq_control_direct.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_direct, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )

        ok = daq_control_gateway.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        assert ok is True
        assert wait_hashpipe_stopped(daq_control_gateway, run_params["data_dir"]), (
            "hashpipe did not stop within timeout"
        )

        _, status = daq_control_direct.StatusDaq({
            "data_dir":               run_params["data_dir"],
            "check_hashpipe_running": True,
            "check_disk_usage":       False,
            "check_run_dirs":         False,
        })
        assert status.get("hashpipe_running") is False

    def test_gateway_double_start_rejected(self, daq_control_gateway, run_params, ensure_clean_daq_state):
        """A second StartDaq via gateway while hashpipe is running raises ValueError."""
        daq_control_gateway.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_gateway, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        with pytest.raises(ValueError):
            daq_control_gateway.StartDaq(run_params)

    def test_gateway_cleanup_after_stop(self, daq_control_gateway, run_params, ensure_clean_daq_state):
        """CleanupData via gateway succeeds after StopDaq."""
        daq_control_gateway.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_gateway, run_params["data_dir"]), (
            "hashpipe did not start within timeout"
        )
        daq_control_gateway.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
        assert wait_hashpipe_stopped(daq_control_gateway, run_params["data_dir"]), (
            "hashpipe did not stop within timeout"
        )
        ok = daq_control_gateway.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })['success']
        assert ok is True
