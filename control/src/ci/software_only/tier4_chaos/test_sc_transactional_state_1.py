import os
import pathlib
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient

from ci.fixtures.state_probe import StateProbe
from ci.software_only.qa_utils import get_isolated_env


class TestPartialStartRollback:
    @pytest.mark.asyncio
    async def test_when_one_node_fails_during_start_then_all_nodes_halted(
        self,
        daq_control_direct: DaqControlClient,
        daq_control_node2: DaqControlClient,
        run_params: dict,
        state_probe: StateProbe,
        mock_workspace: Path,
    ) -> None:
        """
        Intent: Ensure that a single node failure doesn't leave the rest of the
               observatory in an inconsistent 'orphaned' state.
        Scenario: Start succeeds on node-0 but fails on node-1.
        Assertion: Rollback is triggered; node-0 is stopped, all run directories
                   are cleaned up, and a post-mortem snapshot is written to state/snapshots/.
        """
        import unittest.mock
        from ipaddress import IPv4Address
        from typing import Any as AnyT

        from panoseti_grpc.daq_control.client import AsyncDaqControlClient as _DaqClient

        import control.start as start
        from control.utils import config_file
        from control.utils import util as _util
        from control.utils.pydantic_config_models import DaqNode
        from control.utils.run_state import NodeReceipt, RunStateManager

        # mock_workspace already isolates PSETI_STATE and creates standard subdirs
        RunStateManager().clear_state()

        obs_config = config_file.get_obs_config()
        daq_config = config_file.get_daq_config()
        daq_config.head_node_ip_addr = IPv4Address(f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5')
        daq_config.head_node_data_dir = str(PanoPaths.tmp_dir() / "head_data")
        
        quabo_uids = config_file.get_quabo_uids()
        data_config = config_file.get_data_config()
        network_config = config_file.get_network_config()

        # Add a second node that will be made to fail.
        daq_config.daq_nodes.append(
            DaqNode(
                ip_addr=IPv4Address("192.168.0.20"),
                data_dir=str(PanoPaths.tmp_dir() / "daq_data"),
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
            tx: AnyT,
            **kwargs: AnyT
        ) -> None:
            """Actually start hashpipe on node-0, write receipt, then fail for node-1."""
            # Track that we are attempting this node, so rollback knows to stop it.
            tx.nodes_attempted.add(str(daq_cfg.daq_nodes[0].ip_addr))

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

        def mock_make_run_dirs(run_nm: str, obs_cfg: AnyT, daq_cfg: AnyT, quabo_uids: AnyT, data_cfg: AnyT, network_cfg: AnyT) -> None:
            """Create the local run directory so archiving logic in rollback finds it."""
            path = Path(daq_cfg.head_node_data_dir) / run_nm
            path.mkdir(parents=True, exist_ok=True)
            # Add a dummy file so Ladder Step 5 (Archive) actually finds something to snapshot
            (path / "dummy_artifact.txt").write_text("test artifact")

        with unittest.mock.patch("control.start.start_recording", mock_start_recording), \
             unittest.mock.patch("control.start.ph_baseline_file_ok", return_value=True), \
             unittest.mock.patch("control.start._check_daq_reachability"), \
             unittest.mock.patch("control.start.start_data_flow"), \
             unittest.mock.patch("control.start.make_run_dirs", side_effect=mock_make_run_dirs), \
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
        aborted_root = pathlib.Path(daq_config.head_node_data_dir) / "_aborted"
        if not aborted_root.exists():
            pytest.fail(
                f"FAIL (SC-022): _aborted/ directory does not exist at {aborted_root} — "
                "start.py never creates post-mortem snapshots on failure.\n"
                "Fix: on any StartDaq rollback, create "
                "<head_node_data_dir>/_aborted/<run_name>/start_failure_context.json"
            )
        snapshots = list(aborted_root.iterdir())
        assert snapshots, f"No aborted snapshots found in {aborted_root}"
        
        latest = max(snapshots, key=lambda p: p.stat().st_mtime)
        assert (latest / "start_failure_context.json").exists(), \
            "Post-mortem snapshot missing start_failure_context.json"


class TestConcurrentStartLocking:
    def test_when_two_concurrent_starts_then_exactly_one_succeeds(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe, 
        mock_workspace: Path,
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

        # mock_workspace isolates PSETI_STATE
        env = get_isolated_env()
        env["TEST_HEAD_DATA_DIR"] = str(PanoPaths.tmp_dir() / "head_data")
        
        subprocess.run(["python3", "-m", "control.stop", "--yes", "--no-collect"], capture_output=True, env=env)
        mgr = RunStateManager()
        mgr.clear_state()
        if mgr.lock_path.exists():
            mgr.lock_path.unlink()

        wrapper_script = """
import sys
import os
import asyncio
import json
from unittest.mock import patch

import control.start as start
from control.utils import util, config_file

async def main():
    sys.argv = ["start.py", "--no-data", "--no-redis", "--no-hv"]

    original_get_daq_config = config_file.get_daq_config
    def mock_get_daq_config():
        import json
        import os
        from control.utils.paths import PanoPaths

        cfg = original_get_daq_config()
        cfg.head_node_data_dir = os.environ.get("TEST_HEAD_DATA_DIR", "/tmp")
        cfg.head_node_ip_addr = f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'
        cfg.head_node_container = True

        # Ensure data directory exists
        os.makedirs(cfg.head_node_data_dir, exist_ok=True)

        # Ensure PH baseline exists
        ph_path = PanoPaths.calibration_file("quabo_ph_baseline.json")
        ph_path.parent.mkdir(parents=True, exist_ok=True)
        if not ph_path.exists():
            with open(ph_path, "w") as f:
                json.dump({"date": "2024-01-01T00:00:00", "quabos": []}, f)

        # Ensure coherence by assigning all modules in obs_config to the first DAQ node
        # FIRST: clear other nodes to avoid overlaps
        for i in range(1, len(cfg.daq_nodes)):
            cfg.daq_nodes[i].module_ids = []

        obs = config_file.get_obs_config()
        from control.utils.config_file import ip_addr_to_module_id
        mids = []
        for dome in obs.domes:
            for mod in dome.modules:
                mids.append(ip_addr_to_module_id(str(mod.ip_addr)))
        if cfg.daq_nodes:
            cfg.daq_nodes[0].module_ids = mids
        # Write matching quabo_uids.json to disk so start.py process is coherent
        uids_path = PanoPaths.config_dir() / "quabo_uids.json"
        uids_dict = {"domes": [{"num": 0, "modules": []}]}
        for dome in obs.domes:
            for module in dome.modules:
                mid = config_file.ip_addr_to_module_id(str(module.ip_addr))
                uids_dict["domes"][0]["modules"].append({
                    "id": mid,
                    "ip_addr": str(module.ip_addr),
                    "quabos": [{"uid": f"q{mid}_{j}"} if j==0 else {"uid": ""} for j in range(4)]
                })
        with open(uids_path, "w") as f:
            json.dump(uids_dict, f)

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
                f"FAIL (SC-024): expected exactly 1 winner, got {len(winners)}.\\n"
                f"RC1: {rc1}\\nOut1: {out1}\\n"
                f"RC2: {rc2}\\nOut2: {out2}\\n"
            )
        finally:
            if os.path.exists("tmp_start_wrapper.py"):
                os.remove("tmp_start_wrapper.py")
            # Cleanup
            subprocess.run(["python3", "-m", "control.stop", "--yes", "--no-collect"], capture_output=True, env=env)
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
        self.test_when_two_concurrent_starts_then_exactly_one_succeeds(
            daq_control_direct, run_params, state_probe, tmp_path
        )
