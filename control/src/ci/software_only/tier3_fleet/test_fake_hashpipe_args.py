"""
test_fake_hashpipe_args.py — Fleet test verifying argument forwarding to fake_hashpipe.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from ci.software_only.infra.spec import FleetSpec
from ci.software_only.orchestrator.fleet import Fleet
from ci.software_only.tier1_unit.daq_config_fixtures import VALID_CONFIGS
from ci.software_only.tier3_fleet.conftest import make_startdaq_params, requires_docker
from control.driver.quabo_driver import get_daq_params
from control.utils.pydantic_config_models import DataConfig

pytestmark = pytest.mark.tier3

SPEC_SINGLE_NODE = (
    FleetSpec(seed=101, name="hashpipe_args_test", tier="tier3")
    .with_headnode(ip="10.0.1.5")
    .add_dome("d0", lat=37, lon=-121, alt=1000)
    .add_module(200, ip="192.168.3.32")
    .add_daq_node(ip="192.168.0.10", modules=[200], bindhost="lo")
)


def wait_until(
    condition: Any,
    timeout: float = 10.0,
    interval: float = 0.2,
) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if condition():
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


@requires_docker
@pytest.mark.parametrize(
    "pseti_workspace_session",
    [SPEC_SINGLE_NODE],
    indirect=True,
)
@pytest.mark.timeout(120)
class TestFakeHashpipeArgs:
    """Verifies that StartDaq correctly forwards arguments to the hashpipe process."""

    @pytest.mark.parametrize("config_dict", VALID_CONFIGS)
    def test_when_daq_started_then_fake_hashpipe_receives_correct_args(
        self, session_fleet: Fleet, config_dict: dict
    ) -> None:
        fleet = session_fleet
        client = fleet.daq_control_client(0)
        
        # 1. Prepare configuration models
        data_config = DataConfig(**config_dict)
        daq_params = get_daq_params(data_config)
        
        run_dir = f"run_{data_config.run_type}.pffd"
        
        # 2. Map DataConfig to StartDaq params
        # Note: in a real start.py, these are extracted and passed to the RPC.
        start_params = make_startdaq_params(
            fleet, 0, run_dir,
            max_file_size_mb=data_config.max_file_size_mb or 1000.0,
            group_ph_frames=daq_params.do_group_ph_frames,
            obs=data_config.run_type
        )

        # 3. Start DAQ
        ok = client.StartDaq(start_params)
        assert ok is True

        # 4. Wait for hashpipe to start and write its args file
        def check_running() -> bool:
            _, s = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
            return bool(s["hashpipe_running"])

        assert wait_until(check_running), "Hashpipe did not start"

        # 5. Read and verify the recorded arguments
        # Path: /data/module_200/{run_dir}/fake_hashpipe_args.json
        args_file_path = f"/data/module_200/{run_dir}/fake_hashpipe_args.json"
        
        def read_args() -> dict | None:
            ec, out = fleet.exec_in_node(0, f"cat {args_file_path}")
            if ec == 0:
                return json.loads(out)
            return None

        recorded_args = wait_until(lambda: read_args() is not None)
        assert recorded_args is True
        
        args = read_args()
        assert args is not None
        
        options = args["options"]
        assert options["BINDHOST"] == start_params["bindhost"]
        assert float(options["MAXFILESIZE"]) == float(start_params["max_file_size_mb"])
        # bool is stringified in cmd line as integer: "1" or "0"
        assert options["GROUPPHFRAMES"] == str(int(start_params["group_ph_frames"]))
        assert options["RUNDIR"] == start_params["run_dir"]
        assert options["OBS"] == start_params["obs"]
        assert "module.config" in options["CONFIG"]

        # Cleanup for next iteration
        client.StopDaq({"data_dir": "/data", "run_dir": run_dir})
        client.CleanupData({"data_dir": "/data", "run_dir": run_dir, "module_id": [200]})
        client.close()

    def test_when_invalid_params_sent_then_start_daq_fails(
        self, session_fleet: Fleet
    ) -> None:
        """Verifies that the server-side Pydantic model blocks invalid requests."""
        client = session_fleet.daq_control_client(0)
        
        # Invalid: bindhost too long (> 16 chars)
        bad_params = make_startdaq_params(session_fleet, 0, "fail_run")
        bad_params["bindhost"] = "this_interface_name_is_way_too_long_for_pydantic"
        
        with pytest.raises(ValueError) as excinfo:
             client.StartDaq(bad_params)
        
        assert "validation error" in str(excinfo.value).lower()
        assert "bindhost" in str(excinfo.value)
        client.close()
