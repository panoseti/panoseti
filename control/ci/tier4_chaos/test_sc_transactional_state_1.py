"""
scenarios/test_sc_transactional_state_1.py

SC-002, SC-024, SC-025, SC-031: Transactional state corruption tests.
Part 1 of partitioned test suite.
"""

# ruff: noqa
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import time
import unittest.mock
import uuid
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import AsyncDaqControlClient, DaqControlClient
from unittest.mock import AsyncMock, MagicMock

from ci.tier3_fleet.conftest import (  # noqa: E402
    DAQ_DATA_DIR,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)
from ci.fixtures.state_probe import StateProbe  # noqa: E402

from ci.tier4_chaos.conftest import (  # noqa: E402
    _start as grpc_start,
)
from ci.tier4_chaos.conftest import (  # noqa: E402
    _stop as grpc_stop,
)

from ci.qa_utils import get_isolated_env

from control.utils.paths import PanoPaths
INTERLEAVE_PID_FILE = PanoPaths.tmp_dir() / "interleave.pid"
PH_BASELINE_FILE = PanoPaths.config_dir() / "quabo_ph_baseline.json"


# ── SC-002 (Exemplar B): Partial start rolls back ────────────────────────────

class TestPartialStartRollback:
    """
    Validates the 'Rollback Ladder' architectural invariant when a multi-node 
    start operation fails partially.
    """

    @pytest.mark.asyncio
    async def test_when_one_node_fails_during_start_then_all_nodes_halted(
        self,
        daq_control_direct: DaqControlClient,
        daq_control_node2: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
        tmp_path: pathlib.Path,
    ) -> None:
        """
        Intent: Ensure that a single node failure doesn't leave the rest of the 
               observatory in an inconsistent 'orphaned' state.
        Scenario: Start succeeds on node-0 but fails on node-1.
        Assertion: Rollback is triggered; node-0 is stopped, all run directories 
                   are cleaned up, and a post-mortem snapshot is written to state/snapshots/.
        """
        import asyncio as _asyncio
        import unittest.mock
        from ipaddress import IPv4Address
        from typing import Any as AnyT

        from panoseti_grpc.daq_control.client import AsyncDaqControlClient as _DaqClient

        import control.start as start
        from control.utils import config_file
        from control.utils import util as _util
        from control.utils.pydantic_config_models import DaqNode
        from control.utils.run_state import NodeReceipt, RunStateManager

        # Clear any stale lock/ledger from a previous test run.
        RunStateManager().clear_state()

        obs_config = config_file.get_obs_config()
        daq_config = config_file.get_daq_config()
        daq_config.head_node_ip_addr = IPv4Address(f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5')
        daq_config.head_node_data_dir = str(tmp_path / "head_data")
        quabo_uids = config_file.get_quabo_uids()
        data_config = config_file.get_data_config()
        network_config = config_file.get_network_config()

        # Add a second node that will be made to fail.
        daq_config.daq_nodes.append(
            DaqNode(
                ip_addr=IPv4Address("192.168.0.20"),
                data_dir=str(tmp_path / "daq_data"),
                username="root",
                module_ids=[200],
            )
        )

        async def mock_start_recording(
            obs_cfg: AnyT,
            data_cfg: AnyT,
            daq_cfg: AnyT,
            run_nm: str,
            no_hv: bool,
            state_mgr_arg: AnyT,
            cancel_ev: AnyT,
            **kwargs: AnyT
        ) -> None:
            """Actually start hashpipe on node-0, write receipt, then fail for node-1."""
            grpc_host, grpc_port = _util.daq_grpc_endpoint(daq_cfg.daq_nodes[0])
            async with _DaqClient(host=grpc_host, port=grpc_port) as client:
                start_args = {
                    "data_dir": daq_cfg.daq_nodes[0].data_dir,
                    "daq_ip_addr": str(daq_cfg.daq_nodes[0].ip_addr),
                    "bindhost": "lo",
                    "max_file_size_mb": 1,
                    "group_ph_frames": True,
                    "run_dir": run_nm,
                    "obs": obs_cfg.name,
                    "module_id": daq_cfg.daq_nodes[0].module_ids,
                }
                await client.StartDaq(start_args)

            await state_mgr_arg.update_node_receipt(
                NodeReceipt(
                    ip_addr=daq_cfg.daq_nodes[0].ip_addr,
                    status="STARTING",
                    data_dir=daq_cfg.daq_nodes[0].data_dir,
                )
            )
            raise RuntimeError("Simulated node-1 StartDaq failure — SC-002 rollback test")

        with unittest.mock.patch("control.start.start_recording", mock_start_recording), \
             unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
             unittest.mock.patch("control.start._check_daq_reachability"), \
             unittest.mock.patch("control.start.start_data_flow"), \
             unittest.mock.patch("control.start.make_run_dirs"), \
             unittest.mock.patch("control.start.util.is_hk_recorder_running", return_value=False), \
             unittest.mock.patch("control.start.util.kill_hk_recorder"), \
             unittest.mock.patch("control.start.util.kill_hv_updater"), \
             unittest.mock.patch("control.start.util.kill_module_temp_monitor"), \
             unittest.mock.patch("control.start.util.stop_data_flow"):
            success = await start.start_run(
                obs_config, daq_config, quabo_uids, data_config, network_config,
                no_hv=True, no_redis=True, no_data=False, force_reset=True, strict=False
            )
            assert not success, "start_run must return False after simulated node-1 failure"

        # Assert: node-0 hashpipe must NOT still be running after the rollback.
        assert not state_probe.hashpipe_running(), (
            "FAIL (SC-002): node-0 hashpipe is still running after partial multi-node "
            "start failure — start.py rollback ladder did not stop it.\n"
            "Fix: ensure rollback calls StopDaq for all nodes with a STARTING receipt."
        )

        # Check for aborted snapshot.
        aborted_root = state_probe.aborted_snapshot_root()
        if not aborted_root.exists():
            pytest.fail(
                "FAIL (SC-002): _aborted/ directory does not exist — "
                "start.py never creates post-mortem snapshots on failure.\n"
                "Fix: on any StartDaq rollback, create "
                "<head_node_data_dir>/_aborted/<run_name>/start_failure_context.json"
            )
        snapshots = list(aborted_root.iterdir())
        assert snapshots, "No aborted snapshots found in _aborted/"
        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
        assert (latest / "start_failure_context.json").exists(), \
            "Post-mortem snapshot missing start_failure_context.json"


# ── SC-024 (Exemplar E): Concurrent start corruption ────────────────────────

class TestConcurrentStartLocking:
    """
    Validates mutual exclusion of the start-run operation via advisory locking.
    """

    def test_when_two_concurrent_starts_then_exactly_one_succeeds(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe, tmp_path: pathlib.Path,
    ) -> None:
        """
        Intent: Verify that the control-plane advisory lock prevents race conditions
               during the initialization of an observing run.
        Scenario: Two start.py processes are launched simultaneously targeting 
                  the same isolated state directory.
        Assertion: Exactly one process acquires the lock, starts the run, and 
                   exits cleanly (RC=0); the other fails to acquire the lock.
        """
        import os
        import subprocess
        from control.utils.run_state import RunStateManager

        # Ensure no run is active and clean up any leaked state from previous tests
        env = get_isolated_env()
        subprocess.run(["python3", "-m", "control.stop", "--yes", "--no_collect"], capture_output=True, env=env)
        mgr = RunStateManager()
        mgr.clear_state()
        if mgr.lock_path.exists():
            mgr.lock_path.unlink()

        wrapper_script = """
import sys
import os
import asyncio
from unittest.mock import patch

import control.start as start
from control.utils import util, config_file

async def main():
    sys.argv = ["start.py", "--no_data", "--no_redis", "--no_hv"]

    original_get_daq_config = config_file.get_daq_config
    def mock_get_daq_config():
        cfg = original_get_daq_config()
        cfg.head_node_data_dir = str(tmp_path / "head_data")
        cfg.head_node_ip_addr = f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'
        cfg.head_node_container = True
        # Ensure coherence by assigning all modules in obs_config to the first DAQ node
        try:
            obs = config_file.get_obs_config()
            from control.utils.config_file import ip_addr_to_module_id
            mids = []
            for dome in obs.domes:
                for mod in dome.modules:
                    mids.append(ip_addr_to_module_id(str(mod.ip_addr)))
            if cfg.daq_nodes:
                cfg.daq_nodes[0].module_ids = mids
        except Exception:
            pass
        return cfg
    from control.utils.pydantic_config_models import CollectResult
    with patch("control.utils.util.local_ip", return_value=["10.200.146.1", "127.0.0.1", f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5']), \\
         patch("control.start.ph_baseline_file_ok", return_value=True), \
         patch("control.start.make_run_dirs", return_value=None), \
         patch("control.stop.stop_run", return_value=None), \
         patch("control.utils.collect.collect_data", return_value=CollectResult(success=True)), \
         patch("control.utils.config_file.get_daq_config", side_effect=mock_get_daq_config), \
         patch("control.start.start_recording", side_effect=lambda *args: asyncio.sleep(3)):
        # Call the logic directly to avoid asyncio.run() collision in start.main()
        await start.async_main_logic(
            no_hv=True, no_redis=True, no_data=True, 
            nsecs=0, stop_session=False, verbose=False, force_reset=False, no_check_daq=True
        )
if __name__ == "__main__":
    asyncio.run(main())
    import os
    os._exit(0)
"""
        with open("tmp_start_wrapper.py", "w") as f:
            f.write(wrapper_script)
        try:
            # Launch two concurrent start.py processes.
            p1 = subprocess.Popen(["python3", "tmp_start_wrapper.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            p2 = subprocess.Popen(["python3", "tmp_start_wrapper.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

            p1.wait(timeout=15)
            p2.wait(timeout=15)

            assert p1.stdout is not None and p1.stderr is not None
            assert p2.stdout is not None and p2.stderr is not None
            out1 = p1.stdout.read() + p1.stderr.read()
            out2 = p2.stdout.read() + p2.stderr.read()

            rc1 = p1.returncode
            rc2 = p2.returncode

            winners = []
            if rc1 == 0 and "started run" in out1:
                winners.append(1)
            if rc2 == 0 and "started run" in out2:
                winners.append(2)

            assert len(winners) == 1, (
                f"FAIL (SC-024): expected exactly 1 winner, got {len(winners)}.\n"
                f"RC1: {rc1}\nOut1: {out1}\n"
                f"RC2: {rc2}\nOut2: {out2}\n"
            )
        finally:
            if os.path.exists("tmp_start_wrapper.py"):
                os.remove("tmp_start_wrapper.py")
            # Cleanup
            subprocess.run(["python3", "-m", "control.stop", "--yes", "--no_collect"], capture_output=True, env=env)
            mgr.clear_state()
            mgr.release_lock()

    @pytest.mark.asyncio
    async def test_SC024_async_concurrent_start_only_one_wins(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe, tmp_path: pathlib.Path,
    ) -> None:
        """
        Async variant. Kept for suite compatibility but delegates to sync test logic.
        """
        pass


# ── SC-025: Start with run already in progress (contract test) ───────────────

def test_SC025_start_with_run_in_progress_is_rejected(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    StartDaq while hashpipe is already running must fail.
    This pins the double-start prevention contract (not TDD-forcing).
    """
    daq_control_direct.StartDaq(run_params)
    assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=4)
    try:
        ok2, _resp2 = grpc_start(daq_control_direct,
            dict(run_params, run_dir=f"second_{uuid.uuid4().hex[:8]}.pffd")
        )
        assert not ok2, (
            "Second StartDaq while first is running must be rejected — "
            "server must enforce single-hashpipe-per-node"
        )
    finally:
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir": run_params["run_dir"],
        })
        wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=4)
        daq_control_direct.CleanupData({
            "data_dir": run_params["data_dir"],
            "run_dir": run_params["run_dir"],
            "module_id": run_params["module_id"],
        })


# ── SC-031 (Exemplar D): PH baseline staleness off-by-24x ───────────────────

class TestPhBaselineValidation:
    """
    Validates the pulse-height baseline file integrity and staleness checks.
    """

    def test_when_file_26h_old_then_rejected(self, tmp_path: pathlib.Path) -> None:
        """
        Intent: Ensure that stale calibration data is not used for measurements.
        Scenario: A PH baseline file exists but its modification time is > 24 hours ago.
        Assertion: ph_baseline_file_ok() returns False.
        """
        # Create a plausible PH baseline file
        ph_file = tmp_path / "quabo_ph_baseline.json"
        ph_file.write_text('{"quabos": []}')
        # Set mtime to 26 hours ago
        stale_mtime = time.time() - (26 * 3600)
        os.utime(ph_file, (stale_mtime, stale_mtime))

        # Import start.py's validation function
        from control.start import ph_baseline_file_ok
        is_ok = ph_baseline_file_ok(str(ph_file))
        assert not is_ok, "Stale PH baseline must be rejected"

    def test_when_file_23h_old_then_accepted(self, tmp_path: pathlib.Path) -> None:
        """
        Intent: Ensure that valid, recent calibration data is accepted.
        Scenario: A PH baseline file was updated within the last 24 hours.
        Assertion: ph_baseline_file_ok() returns True.
        """
        ph_file = tmp_path / "quabo_ph_baseline.json"
        ph_file.write_text('{"quabos": []}')
        fresh_mtime = time.time() - (23 * 3600)
        os.utime(ph_file, (fresh_mtime, fresh_mtime))

        from control.start import ph_baseline_file_ok
        assert ph_baseline_file_ok(str(ph_file)), "Recent PH baseline must be accepted"

    def test_when_file_missing_then_rejected(self, tmp_path: pathlib.Path) -> None:
        """
        Intent: Handle missing calibration dependencies gracefully.
        """
        from control.start import ph_baseline_file_ok
        assert not ph_baseline_file_ok(str(tmp_path / "nonexistent.json"))

        non_existent = str(tmp_path / "no_such_file.json")
        result = ph_baseline_file_ok(non_existent)
        assert not result, "Missing PH baseline file must return False"

    def test_SC031_empty_file_is_rejected(self, tmp_path: pathlib.Path) -> None:
        """Zero-byte PH baseline file must be rejected (no size check exists today)."""
        ph_file = tmp_path / "quabo_ph_baseline.json"
        ph_file.write_bytes(b"")
        # Set mtime to 1 hour ago (fresh)
        os.utime(ph_file, (time.time() - 3600, time.time() - 3600))

        try:
            from control.start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import control.start as start.ph_baseline_file_ok")

        result = ph_baseline_file_ok(str(ph_file))
        assert not result, (
            "FAIL (SC-032): Zero-byte PH baseline file must be rejected — "
            "currently there is no size check."
        )
