"""
scenarios/test_sc_transactional_state_4.py

SC-026, SC-027, SC-029, SC-030: Transactional state corruption tests.
Part 4 of partitioned test suite.
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

from ci.software_only.conftest import (  # noqa: E402
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)
from ci.software_only.tier3_fleet.conftest import (  # noqa: E402
    DAQ_DATA_DIR,
)
from ci.fixtures.state_probe import StateProbe  # noqa: E402

from ci.software_only.tier4_chaos.conftest import (  # noqa: E402
    _start as grpc_start,
)
from ci.software_only.tier4_chaos.conftest import (  # noqa: E402
    _stop as grpc_stop,
)

from ci.software_only.qa_utils import get_isolated_env

from control.utils.paths import PanoPaths
PH_BASELINE_FILE = PanoPaths.config_dir() / "quabo_ph_baseline.json"


# ── Shared Helpers ───────────────────────────────────────────────────────────

@contextlib.contextmanager
def mock_daq_config_for_headnode():
    """Temporarily patch daq_config.json to point to localhost (CI headnode)."""
    import json

    from control.utils import config_file
    
    from control.utils.paths import PanoPaths
    path = PanoPaths.config_dir() / "daq_config.json"
    backup = str(path) + ".bak"
    # Ensure tmp/ and configs/ exist (should already, but let's be safe)
    PanoPaths.ensure_dirs()
    PanoPaths.config_dir().mkdir(parents=True, exist_ok=True)
    
    # Create a dummy PH baseline if missing
    ph_baseline = PanoPaths.tmp_dir() / "quabo_ph_baseline.json"
    if not os.path.exists(ph_baseline):
        with open(ph_baseline, "w") as f:
            json.dump({"date": "2024-01-01T00:00:00", "quabos": []}, f)

    if os.path.exists(path):
        import shutil
        shutil.copyfile(path, backup)
    
    with open(path) as f:
        cfg = json.load(f)
    
    import tempfile
    # Prefer the isolated HEAD_DATA_DIR set by auto_isolate so the subprocess
    # env and the daq_config.json written here share the same path.  Fall back
    # to a fresh tempdir only when running outside the pytest harness.
    tmp_data_dir = os.environ.get("HEAD_DATA_DIR") or tempfile.mkdtemp()
    os.makedirs(tmp_data_dir, exist_ok=True)

    tester_ip = f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'
    cfg["head_node_ip_addr"] = tester_ip
    cfg["head_node_data_dir"] = tmp_data_dir
    cfg["head_node_container"] = True
    
    # Coherence Fix: Ensure the DAQ node is handling ALL modules defined in the 
    # current obs_config.json to prevent "no DAQ node is handling module X" errors.
    mids = []
    obs = config_file.get_obs_config()
    for dome in obs.domes:
        for module in dome.modules:
            mids.append(config_file.ip_addr_to_module_id(str(module.ip_addr)))

    # Assign ALL modules to the single available CI node
    # Use a DAQ IP that is on the same /24 subnet as the modules (192.168.3.x)
    # to pass strict Tier-2 Subnet Coherence validation.
    daqnode_ip = "192.168.3.30"
    cfg["daq_nodes"] = [
        {
            "ip_addr": daqnode_ip,
            "data_dir": "/data",
            "username": "root",
            "module_ids": mids,
            "bindhost": "lo"
        }
    ]
    
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    
    # Write matching quabo_uids.json to tmp/ so associate() in subprocess passes
    uids_path = PanoPaths.tmp_dir() / "quabo_uids.json"
    uids_path.parent.mkdir(parents=True, exist_ok=True)
    from control.utils.pydantic_config_models import QuaboUids
    uids_dict: dict[str, Any] = {"domes": [{"num": 0, "modules": []}]}
    for mid in mids:
        uids_dict["domes"][0]["modules"].append({
            "id": mid,
            "ip_addr": f"192.168.3.{mid}",
            "quabos": [{"uid": f"q{mid}_{j}"} if j==0 else {"uid": ""} for j in range(4)]
        })
    with open(uids_path, "w") as f:
        json.dump(uids_dict, f, indent=4)

    with unittest.mock.patch("control.utils.config_file.get_quabo_uids", return_value=QuaboUids(**uids_dict)):
        try:
            yield
        finally:
            if os.path.exists(backup):
                import shutil
                shutil.move(backup, path)


# ── SC-026: stop.py with no run in progress ──────────────────────────────────

def test_SC026_stop_with_no_run_is_noop(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
) -> None:
    """
    SC-026: Calling StopDaq when no hashpipe is running must complete cleanly
    and not raise. Pins the no-run-in-progress contract.

    Not TDD-forcing — current behavior: returns success (no-op).
    """
    # Ensure no hashpipe is running
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=5)

    ok, resp = grpc_stop(daq_control_direct, {
        "data_dir": run_params["data_dir"],
        "run_dir": run_params["run_dir"],
    })
    # Must succeed (idempotent) or at least not raise
    assert ok is True or (not ok and resp), (
        "StopDaq with no active run must be a no-op (ok=True) "
        "or return a clear explanation if it returns ok=False"
    )


# ── SC-027: stop.py --run X when current_run says Y ──────────────────────────

class TestSC027StopRunMismatch:
    """
    SC-027: stop_run called with --run X when ledger has run Y must refuse
    (return early) unless force_cleanup=True.

    Pins the mismatch guard at stop.py:~430-437.
    """

    def test_SC027_mismatch_without_force_skips_stop_recording(self, tmp_path: pathlib.Path) -> None:
        """
        stop_run with mismatching run name and force_cleanup=False must
        return early without calling stop_recording.
        """
        import asyncio
        from ipaddress import IPv4Address
        from unittest.mock import AsyncMock, MagicMock, patch

        import control.stop as stop_module
        from control.utils.pydantic_config_models import (
            DaqConfig,
            NetworkConfig,
            QuaboUids,
            RunStateLedger,
        )

        daq_config = DaqConfig(
            head_node_ip_addr=IPv4Address(f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'),
            head_node_data_dir=str(tmp_path / "head_data"),
            daq_nodes=[],
        )
        network_config = NetworkConfig()
        quabo_uids = QuaboUids(domes=[])

        mock_mgr = MagicMock()
        ledger = RunStateLedger(
            run_name="active_run_Y.pffd",
            status="ACTIVE",
            start_time="2026-01-01T00:00:00Z",
        )
        mock_mgr.load_state.return_value = ledger

        mock_stop_rec = AsyncMock()

        with patch("control.stop.RunStateManager", return_value=mock_mgr), \
             patch("socket.gethostbyname", return_value=f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'), \
             patch("control.utils.util.local_ip", return_value=[f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5']), \
             patch("control.stop.stop_recording", mock_stop_rec):

            asyncio.run(stop_module.stop_run(
                daq_config, network_config, quabo_uids,
                verbose=False, run="different_run_X.pffd", force_cleanup=False,
            ))

        assert not mock_stop_rec.called, (
            "FAIL (SC-027): stop_recording was called despite run name mismatch. "
            "The guard at stop.py:~430 (refuse unless --force-cleanup) is missing."
        )

    def test_SC027_mismatch_with_force_proceeds_to_stop_recording(self, tmp_path: pathlib.Path) -> None:
        """
        stop_run with force_cleanup=True must proceed past the mismatch
        guard and call stop_recording.
        """
        import asyncio
        from ipaddress import IPv4Address
        from unittest.mock import AsyncMock, MagicMock, patch

        import control.stop as stop_module
        from control.utils.pydantic_config_models import (
            DaqConfig,
            NetworkConfig,
            QuaboUids,
            RunStateLedger,
        )

        daq_config = DaqConfig(
            head_node_ip_addr=IPv4Address(f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'),
            head_node_data_dir=str(tmp_path / "head_data"),
            daq_nodes=[],
        )
        network_config = NetworkConfig()
        quabo_uids = QuaboUids(domes=[])

        mock_mgr = MagicMock()
        ledger = RunStateLedger(
            run_name="active_run_Y.pffd",
            status="ACTIVE",
            start_time="2026-01-01T00:00:00Z",
        )
        mock_mgr.load_state.return_value = ledger

        mock_stop_rec = AsyncMock()

        with patch("control.stop.RunStateManager", return_value=mock_mgr), \
             patch("socket.gethostbyname", return_value=f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'), \
             patch("control.utils.util.local_ip", return_value=[f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5']), \
             patch("control.stop.stop_recording", mock_stop_rec), \
             patch("control.utils.util.kill_hv_updater"), \
             patch("control.utils.util.kill_hk_recorder"), \
             patch("control.utils.util.kill_module_temp_monitor"), \
             patch("control.utils.util.stop_data_flow"), \
             patch("control.utils.util.remove_run_name"):

            asyncio.run(stop_module.stop_run(
                daq_config, network_config, quabo_uids,
                verbose=False, run="different_run_X.pffd", force_cleanup=True,
            ))

        assert mock_stop_rec.called, (
            "FAIL (SC-027): stop_recording was NOT called even with force_cleanup=True. "
            "The --force-cleanup escape hatch in stop.py is broken."
        )


# ── SC-029: Fundamental failure skips cleanup ───────────────────────────────

class TestSC029FundamentalFailureSkipsCleanup:
    """
    SC-029: if collect_data fails for a node, stop_run must NOT call CleanupData
    for that node, and MUST NOT write the collect_complete marker.
    """

    @pytest.mark.asyncio
    async def test_SC029_fundamental_failure_skips_cleanup(
        self,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Verify that a fundamental failure passed to StopTransaction:
          - Bypasses TransferQueue().enqueue()
          - Transitions ledger to STOPPED_WITH_ERRORS
        """
        from ipaddress import IPv4Address
        from unittest.mock import MagicMock, patch
        
        import control.stop as stop_module
        from control.utils.pydantic_config_models import (
            DaqConfig,
            DaqNode,
            NetworkConfig,
            QuaboUids,
            RunStateLedger,
        )

        # 1. Setup minimal configs
        head_dir = tmp_path / "data" / "head"
        run_name = "test_run_SC029.pffd"
        run_dir = head_dir / run_name
        run_dir.mkdir(parents=True)

        daq_config = DaqConfig(
            head_node_ip_addr=IPv4Address(f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'),
            head_node_data_dir=str(head_dir),
            daq_nodes=[
                DaqNode(ip_addr=IPv4Address("192.168.0.10"), data_dir=str(tmp_path / "daq_data"), username="root", module_ids=[1]),
            ],
        )
        network_config = NetworkConfig()
        quabo_uids = QuaboUids(domes=[])

        # 2. Mock RunStateManager and Ledger
        mock_mgr = MagicMock()
        ledger = RunStateLedger(
            run_name=run_name,
            status="ACTIVE",
            start_time="2026-01-01T00:00:00Z",
        )
        mock_mgr.load_state.return_value = ledger

        # 3. Mock TransferQueue to verify bypass
        mock_tq = MagicMock()
        
        # 4. We will simulate a fundamental failure by mocking util.local_ip 
        # to raise an exception inside the 'with' block of stop_run.
        
        with patch("control.stop.RunStateManager", return_value=mock_mgr), \
             patch("control.utils.util.local_ip", side_effect=RuntimeError("Fundamental Failure")), \
             patch("control.stop.TransferQueue", return_value=mock_tq), \
             patch("socket.gethostbyname", return_value=f'{os.environ.get("HEAD_NET_PREFIX", "10.0.1")}.5'):

            # stop_run swallows generic exceptions and returns False
            success = await stop_module.stop_run(
                daq_config, network_config, quabo_uids,
                verbose=True, no_cleanup=False, no_collect=False,
                run=run_name
            )
            assert success is False

        # ASSERTIONS
        
        # TransferQueue.enqueue must NOT have been called
        assert not mock_tq.enqueue.called, "TransferQueue.enqueue was called despite fundamental failure!"

        # Ledger should have transitioned to STOPPED_WITH_ERRORS
        mock_mgr.transition.assert_called_with("STOPPED_WITH_ERRORS", last_transfer_error="Fundamental Failure")



# ── SC-030: PH baseline file missing ─────────────────────────────────────────

def test_SC030_missing_ph_baseline_file_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    """
    SC-030: start.py must refuse to start if the PH baseline file does not
    exist. Pins the missing-file contract (not TDD-forcing).
    """
    try:
        from control.start import ph_baseline_file_ok
    except ImportError:
        pytest.skip("Could not import control.start as start.ph_baseline_file_ok")

    non_existent = str(tmp_path / "no_such_file.json")
    result = ph_baseline_file_ok(non_existent)
    assert not result, "Missing PH baseline file must cause ph_baseline_file_ok to return False"
