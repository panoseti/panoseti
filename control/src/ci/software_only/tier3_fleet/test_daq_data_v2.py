"""
test_daq_data_v2.py — Integration tests for the DaqData v2 "Push Forwarder" architecture.
"""
from __future__ import annotations

import time

from ci.software_only.conftest import (
    wait_hashpipe_running,
)


class TestDaqDataV2:
    """Verifies simulator -> forwarder -> server lifecycle in the fleet."""

    def test_v2_forwarder_lifecycle(self, daq_control_direct, run_params, ensure_clean_daq_state, daqnode_container) -> None:
        """Verifies that enabling v2 forwarder results in the process starting and stopping properly."""
        
        # 1. Start DAQ with v2 forwarder enabled
        # The aggregator (headnode) target should be a reachable address.
        # Since we are just testing lifecycle, any reachable address will do.
        target_aggregator = "127.0.0.1:50051"
        
        params = dict(run_params)
        params["enable_v2_forwarder"] = True
        params["headnode_target"] = target_aggregator
        
        daq_control_direct.StartDaq(params)
        assert wait_hashpipe_running(daq_control_direct, params["data_dir"])
        
        # 2. Verify forwarder is running in the container
        exit_code, output = daqnode_container.exec_run("pgrep -f panoseti_grpc.daq_data_v2.forwarder")
        assert exit_code == 0, f"Forwarder process not found: {output.decode()}"
        
        # 3. Stop DAQ
        daq_control_direct.StopDaq({"data_dir": params["data_dir"], "run_dir": params["run_dir"]})
        
        # 4. Verify forwarder is gone
        time.sleep(1.0)
        exit_code, output = daqnode_container.exec_run("pgrep -f panoseti_grpc.daq_data_v2.forwarder")
        assert exit_code != 0, f"Forwarder process leaked after StopDaq: {output.decode()}"
