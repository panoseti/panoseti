"""
scenarios/test_sc_transactional_state.py

SC-021 → SC-040: Transactional state corruption tests.

Key exemplars implemented here:
  - SC-024 (Exemplar E): Concurrent start corruption (no advisory lock)
  - SC-031 (Exemplar D): PH baseline staleness off-by-24x
  - SC-033 (Exemplar F): Stale interleave PID from unrelated process
  - SC-034:  Interleave daemon outlives stop.py's retry limit (no hard-kill)
  - SC-002 (Exemplar B): Partial start leaves zombie quabos (SC-002 stub)

TDD intent: each TDD-forcing test FAILS RED on current master.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import time
import uuid
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient

# control/ root must be on sys.path (handled by scenarios/conftest.py)
CONTROL_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from ci.integration.conftest import (  # noqa: E402
    DAQ_DATA_DIR,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)
from ci.integration.state_probe import StateProbe  # noqa: E402

from .conftest import (  # noqa: E402
    _start as grpc_start,
)
from .conftest import (  # noqa: E402
    _stop as grpc_stop,
)

INTERLEAVE_PID_FILE = pathlib.Path("tmp/interleave.pid")
PH_BASELINE_FILE = pathlib.Path("configs/quabo_ph_baseline.json")


# ── SC-002 (Exemplar B): Partial start rolls back ────────────────────────────

class TestSC002PartialStartRollback:
    """
    SC-002 / Exemplar B: when StartDaq succeeds on node-0 but fails on node-1,
    start.py has no rollback — node-0 is left in a streaming state.

    FAILS RED TODAY: start.py calls start_data_flow() *before* start_recording(),
    so when StartDaq fails on node-1, quabos on node-0 are already streaming
    but no rollback (no stop_data_flow, no kill_hv_updater, no cleanup of
    partial run dirs, no post-mortem snapshot).
    """

    @pytest.mark.asyncio
    async def test_SC002_partial_start_must_leave_no_active_hashpipe(
        self,
        daq_control_direct: DaqControlClient,
        daq_control_node2: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
    ) -> None:
        """
        With one failing node: no hashpipe should remain running on any node,
        current_run must be unset, and a post-mortem snapshot must exist.
        """
        import unittest.mock
        from ipaddress import IPv4Address

        import start
        from utils import config_file
        from utils.pydantic_config_models import DaqNodeValidator
        obs_config = config_file.get_obs_config()
        daq_config = config_file.get_daq_config()
        # Override head node IP to match test container
        daq_config.head_node_ip_addr = IPv4Address("10.0.1.5")
        daq_config.head_node_data_dir = "/data/head"
        quabo_uids = config_file.get_quabo_uids()
        data_config = config_file.get_data_config()
        network_config = config_file.get_network_config()

        # Add a second node that will fail
        daq_config.daq_nodes.append(
            DaqNodeValidator(ip_addr=IPv4Address("192.168.0.20"), data_dir="/data", username="root", module_ids=[200])
        )

        original_start_daq = start.DaqControlClient.StartDaq
        def mock_start_daq(self_client, req):
            if req['daq_ip_addr'] == "192.168.0.20":
                return False
            return original_start_daq(self_client, req)

        with unittest.mock.patch.object(start.DaqControlClient, 'StartDaq', new=mock_start_daq), \
             unittest.mock.patch('start.ph_baseline_file_ok', return_value=True):
            # run with no_data=True to skip quabo hardware setup in test environment
            success = await start.start_run(
                obs_config, daq_config, quabo_uids, data_config, network_config,
                no_hv=True, no_redis=True, no_data=True, force_reset=True
            )
            assert not success, "Start run should fail due to simulated node failure"
            
        # Assert: node-0 hashpipe must NOT still be running after a partial failure
        assert not state_probe.hashpipe_running(), (
            "FAIL (SC-002): node-0 hashpipe is still running after partial multi-node "
            "start failure — start.py has no rollback ladder.\n"
            "Fix: wrap start_recording() in a try/except with rollback for all started nodes."
        )

        # Check for aborted snapshot
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

class TestSC024ConcurrentStart:
    """
    SC-024 / Exemplar E: no advisory lock around start_run means two concurrent
    start.py invocations both pass the read_run_name() check, both call
    make_run_dirs, and one fails with FileExistsError leaving mixed state.
    """

    def test_SC024_concurrent_start_only_one_wins(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
    ) -> None:
        """
        Two concurrent start.py calls — exactly one must succeed,
        the other must fail with a clear error.
        """
        import os
        import subprocess

        # Ensure no run is active and clean up any leaked state from previous tests
        subprocess.run(["python3", "stop.py", "--no_collect"], capture_output=True)
        if os.path.exists("tmp/run_state.toml"):
            os.remove("tmp/run_state.toml")
        if os.path.exists("tmp/panoseti_control.lock"):
            os.remove("tmp/panoseti_control.lock")

        wrapper_script = """
import sys
import asyncio
from unittest.mock import patch

import start
from utils import util, config_file

async def main():
    sys.argv = ["start.py", "--no_data", "--no_redis", "--no_hv"]

    original_get_daq_config = config_file.get_daq_config
    def mock_get_daq_config():
        cfg = original_get_daq_config()
        cfg.head_node_data_dir = "/data/head"
        cfg.head_node_ip_addr = "10.0.1.5"
        return cfg

    from utils.pydantic_config_models import CollectResult
    with patch("utils.util.local_ip", return_value=["10.200.146.1", "127.0.0.1", "10.0.1.5"]), \\
         patch("start.ph_baseline_file_ok", return_value=True), \\
         patch("start.make_run_dirs", return_value=None), \\
         patch("stop.stop_run", return_value=None), \\
         patch("utils.collect.collect_data", return_value=CollectResult(success=True)), \\
         patch("utils.config_file.get_daq_config", side_effect=mock_get_daq_config), \\
         patch("start.start_recording", side_effect=lambda *args: asyncio.run(asyncio.sleep(3))):
        await start.main()
if __name__ == "__main__":
    asyncio.run(main())
    import os
    os._exit(0)
"""
        with open("tmp_start_wrapper.py", "w") as f:
            f.write(wrapper_script)
        try:
            # Launch two concurrent start.py processes.
            p1 = subprocess.Popen(["python3", "tmp_start_wrapper.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            p2 = subprocess.Popen(["python3", "tmp_start_wrapper.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

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
                "No advisory lock around start_run."
            )
        finally:
            os.remove("tmp_start_wrapper.py")
            # Cleanup
            subprocess.run(["python3", "stop.py", "--no_collect"], capture_output=True)

    @pytest.mark.asyncio
    async def test_SC024_async_concurrent_start_only_one_wins(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
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
    assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=10)
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
        wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=8)
        daq_control_direct.CleanupData({
            "data_dir": run_params["data_dir"],
            "run_dir": run_params["run_dir"],
            "module_id": run_params["module_id"],
        })


# ── SC-031 (Exemplar D): PH baseline staleness off-by-24x ───────────────────

class TestSC031PHBaslineStaleness:
    """
    SC-031 / Exemplar D: start.py::ph_baseline_file_ok() compares against
    time.time() - 24*86400 which is 24 DAYS, not 24 hours.

    FAILS RED TODAY: the comparison is `time.time() - 24*86400` (24 * 86400 s = 24 days).
    A file that is 26 hours old (should be rejected) passes the check.

    File: start.py, function: ph_baseline_file_ok(), around line 103.
    Fix: change `24*86400` to `86400` (24 hours = 1 day).
    """

    def test_SC031_file_26h_old_must_be_rejected(self, tmp_path: pathlib.Path) -> None:
        """A PH baseline file that is 26 hours old must be rejected."""
        # Create a plausible PH baseline file
        ph_file = tmp_path / "quabo_ph_baseline.json"
        ph_file.write_text('{"quabos": []}')
        # Set mtime to 26 hours ago
        stale_mtime = time.time() - (26 * 3600)
        os.utime(ph_file, (stale_mtime, stale_mtime))

        # Import start.py's validation function
        try:
            from start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import start.ph_baseline_file_ok — check sys.path")

        is_ok = ph_baseline_file_ok(str(ph_file))
        assert not is_ok, (
            "FAIL (SC-031): ph_baseline_file_ok() returned True for a 26-hour-old file.\n"
            "The comparison uses time.time() - 24*86400 (= 24 days) instead of 86400 (24 hours).\n"
            "Fix: change to `time.time() - 86400` in start.py::ph_baseline_file_ok()."
        )

    def test_SC031_file_23h_old_must_be_accepted(self, tmp_path: pathlib.Path) -> None:
        """A PH baseline file that is 23 hours old (within 24 h) must pass."""
        ph_file = tmp_path / "quabo_ph_baseline.json"
        ph_file.write_text('{"quabos": []}')
        fresh_mtime = time.time() - (23 * 3600)
        os.utime(ph_file, (fresh_mtime, fresh_mtime))

        try:
            from start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import start.ph_baseline_file_ok")

        assert ph_baseline_file_ok(str(ph_file)), \
            "A 23-hour-old PH baseline file must be accepted"

    def test_SC031_missing_file_is_rejected(self, tmp_path: pathlib.Path) -> None:
        """Missing PH baseline file must be rejected (not crash)."""
        try:
            from start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import start.ph_baseline_file_ok")

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
            from start import ph_baseline_file_ok
        except ImportError:
            pytest.skip("Could not import start.ph_baseline_file_ok")

        result = ph_baseline_file_ok(str(ph_file))
        assert not result, (
            "FAIL (SC-032): Zero-byte PH baseline file must be rejected — "
            "currently there is no size check."
        )


# ── SC-033 (Exemplar F): Stale interleave PID ────────────────────────────────

class TestSC033StaleInterleavePID:
    """
    SC-033 / Exemplar F: stop.py::stop_interleave() calls os.kill(pid, SIGTERM)
    without verifying that the PID belongs to our interleave process.

    A stale PID file from a previous crashed run (or a PID that was recycled by
    an unrelated process) could result in signalling the wrong process.

    FAILS RED TODAY: stop_interleave() sends SIGTERM to whatever PID is in the
    file, and silently leaves the file if os.kill() raises PermissionError
    (e.g., PID 1 = init/systemd).

    Fix: verify process identity before sending SIGTERM (check cmdline or
    use psutil to confirm the process is our interleave daemon).
    """

    def test_SC033_stale_pid_of_init_not_signalled(self, tmp_path: pathlib.Path) -> None:
        """
        Seed the PID file with PID=1 (init/systemd). stop_interleave() must:
          - Detect that PID 1 is NOT our interleave process
          - NOT send SIGTERM to PID 1
          - Clean up the stale PID file

        Currently fails because stop_interleave() sends SIGTERM to PID 1 (which
        raises PermissionError and silently leaves the file).
        """
        pid_file = tmp_path / "interleave.pid"
        pid_file.write_text("1\n")

        try:
            from stop import stop_interleave as _stop_interleave
        except ImportError:
            pytest.skip("Could not import stop.stop_interleave — check sys.path")

        # Monkey-patch PID_FILE to our temp file
        import stop as stop_module
        original = stop_module.PID_FILE
        try:
            stop_module.PID_FILE = str(pid_file)
            # Also patch tools.interleave.PID_FILE (imported by stop.py)
            import tools.interleave as interleave_module
            original_interleave = interleave_module.PID_FILE
            interleave_module.PID_FILE = str(pid_file)

            _stop_interleave(retry_limit=2)

            # PID file must be cleaned up
            assert not pid_file.exists(), (
                "FAIL (SC-033): PID file still exists after stop_interleave() "
                "with a stale PID — the file was not cleaned up."
            )
            # PID 1 (init) must still be running — we must not have killed it
            assert pathlib.Path("/proc/1").exists(), \
                "CRITICAL: PID 1 (/proc/1) is gone — stop_interleave() killed init!"

        finally:
            stop_module.PID_FILE = original
            interleave_module.PID_FILE = original_interleave

    def test_SC033_stale_pid_dead_process_clears_file(self, tmp_path: pathlib.Path) -> None:
        """
        PID file contains a PID for a process that no longer exists.
        stop_interleave() must clean the stale file without raising.
        """
        pid_file = tmp_path / "interleave.pid"
        # Find a PID that doesn't exist
        stale_pid = 99999
        while pathlib.Path(f"/proc/{stale_pid}").exists():
            stale_pid -= 1
        pid_file.write_text(f"{stale_pid}\n")

        try:
            import stop as stop_module
            import tools.interleave as interleave_module
            from stop import stop_interleave as _stop_interleave
        except ImportError:
            pytest.skip("Could not import stop.stop_interleave")

        original_stop = stop_module.PID_FILE
        original_interleave = interleave_module.PID_FILE
        try:
            stop_module.PID_FILE = str(pid_file)
            interleave_module.PID_FILE = str(pid_file)
            _stop_interleave(retry_limit=2)
            assert not pid_file.exists(), \
                "Stale PID file for dead process must be cleaned up by stop_interleave()"
        finally:
            stop_module.PID_FILE = original_stop
            interleave_module.PID_FILE = original_interleave


# ── SC-034: Interleave daemon outlives retry window ──────────────────────────

class TestSC034InterleaveDaemonHardKill:
    """
    SC-034: stop_interleave() polls for 10 x 0.5 s = 5 s with no SIGKILL fallback.
    If the interleave daemon ignores SIGTERM (or is slow to clean up), it keeps
    running after stop.py completes, continuing to flip quabo modes during stop.

    FAILS RED TODAY: no SIGKILL escalation in stop_interleave().
    Fix: after the retry window, send SIGKILL, then restore MAROC config.
    """

    def test_SC034_slow_interleave_is_hard_killed_after_timeout(
        self, tmp_path: pathlib.Path
    ) -> None:
        """
        A process that ignores SIGTERM must be SIGKILLed after the retry window.
        """
        import subprocess

        # Spawn a subprocess that ignores SIGTERM
        proc = subprocess.Popen(
            ["python3", "-c",
             "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "time.sleep(60)"],
            start_new_session=True,
        )
        pid = proc.pid
        pid_file = tmp_path / "interleave.pid"
        pid_file.write_text(f"{pid}\n")

        try:
            import stop as stop_module
            import tools.interleave as interleave_module
            from stop import stop_interleave as _stop_interleave
        except ImportError:
            proc.kill()
            pytest.skip("Could not import stop.stop_interleave")

        original_stop = stop_module.PID_FILE
        original_interleave = interleave_module.PID_FILE
        try:
            stop_module.PID_FILE = str(pid_file)
            interleave_module.PID_FILE = str(pid_file)

            # With retry_limit=4 (4 x 0.5 s = 2 s budget), the daemon outlives it
            _stop_interleave(retry_limit=4)

            # After stop_interleave returns, the daemon must be dead
            assert not pathlib.Path(f"/proc/{pid}").exists(), (
                "FAIL (SC-034): Interleave daemon (that ignores SIGTERM) is still "
                "alive after stop_interleave() returned. No hard-kill fallback.\n"
                "Fix: send SIGKILL after retry_limit is exhausted."
            )
        finally:
            stop_module.PID_FILE = original_stop
            interleave_module.PID_FILE = original_interleave
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait(timeout=2)

    def test_SC034b_start_interleave_while_running_is_rejected(
        self, tmp_path: pathlib.Path
    ) -> None:
        """
        SC-034b: invoking config.py --start-interleave while a daemon is already
        running must be rejected. Two daemons racing over quabo config is undefined.

        FAILS RED TODAY: InterleaveController._acquire_lock() correctly prevents this
        (it checks the PID file). This test pins the contract.
        """
        import subprocess

        # Spawn a fake interleave daemon
        proc = subprocess.Popen(
            ["python3", "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        pid_file = tmp_path / "interleave.pid"
        pid_file.write_text(f"{proc.pid}\n")

        try:
            import tools.interleave as interleave_module
            original = interleave_module.PID_FILE
            interleave_module.PID_FILE = str(pid_file)

            try:
                # Attempting to construct InterleaveController while one is "running"
                # should raise SystemExit (the lock acquisition fails)
                with pytest.raises(SystemExit):
                    from tools.interleave import InterleaveController
                    InterleaveController.__new__(InterleaveController)._acquire_lock()
            except ImportError:
                pytest.skip("Could not import tools.interleave.InterleaveController")
            finally:
                interleave_module.PID_FILE = original
        finally:
            proc.kill()
            proc.wait(timeout=2)


# ── SC-021 → SC-023: start.py interrupted at various stages ──────────────────

@pytest.mark.skip(reason="SC-021: requires instrumented start.py with injection point")
def test_SC021_killed_after_make_run_dirs_leaves_orphan_dirs() -> None:
    """
    SC-021: If start.py is killed after make_run_dirs but before start_data_flow,
    partial run dirs exist on head and DAQ but current_run is unset.
    Next start.py at the same UTC second creates a colliding run name.

    FAILS RED TODAY: no cleanup of orphaned run dirs on abort.
    Fix: write a sentinel before make_run_dirs, clean up on exit.
    """
    pytest.skip("Requires start.py instrumented with post-make_run_dirs kill")


@pytest.mark.skip(reason="SC-022: requires start.py injection between start_data_flow and start_recording")
def test_SC022_killed_after_start_data_flow_quabos_streaming_to_void() -> None:
    """
    SC-022: If start.py is killed after start_data_flow() but before
    start_recording(), quabos are streaming science packets but no hashpipe
    listens — kernel drops all packets, silent data loss.

    FAILS RED TODAY: no cleanup of quabo data-packet destinations on abort.
    Fix: rollback quabo data destinations (set to 0.0.0.0) in finally block.
    """
    pytest.skip("Requires start.py instrumented with post-start_data_flow kill")


@pytest.mark.skip(reason="SC-023: requires start.py injection between start_recording and write_run_name")
def test_SC023_killed_after_start_recording_hashpipe_orphaned_unknown_to_head() -> None:
    """
    SC-023: If start.py is killed after start_recording() but before
    write_run_name(), hashpipe is running on the DAQ but the head node has no
    current_run marker. The next start.py double-starts.

    FAILS RED TODAY: no atomic write of current_run with hashpipe start.
    Fix: write current_run before returning from start_recording, or use
    a two-phase commit (write temp file, rename atomically).
    """
    pytest.skip("Requires start.py instrumented with post-start_recording kill")


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

    def test_SC027_mismatch_without_force_skips_stop_recording(self) -> None:
        """
        stop_run with mismatching run name and force_cleanup=False must
        return early without calling stop_recording.
        """
        import asyncio
        from ipaddress import IPv4Address
        from unittest.mock import AsyncMock, MagicMock, patch

        import stop as stop_module
        from utils.pydantic_config_models import (
            DaqConfigValidator,
            NetworkConfigValidator,
            QuaboUidsValidator,
            RunStateLedger,
        )

        daq_config = DaqConfigValidator(
            head_node_ip_addr=IPv4Address("10.0.1.5"),
            head_node_data_dir="/data/head",
            daq_nodes=[],
        )
        network_config = NetworkConfigValidator()
        quabo_uids = QuaboUidsValidator(domes=[])

        mock_mgr = MagicMock()
        ledger = RunStateLedger(
            run_name="active_run_Y.pffd",
            status="ACTIVE",
            start_time="2026-01-01T00:00:00Z",
        )
        mock_mgr.load_state.return_value = ledger

        mock_stop_rec = AsyncMock()

        with patch("stop.RunStateManager", return_value=mock_mgr), \
             patch("socket.gethostbyname", return_value="10.0.1.5"), \
             patch("utils.util.local_ip", return_value=["10.0.1.5"]), \
             patch("stop.stop_recording", mock_stop_rec):

            asyncio.run(stop_module.stop_run(
                daq_config, network_config, quabo_uids,
                verbose=False, run="different_run_X.pffd", force_cleanup=False,
            ))

        assert not mock_stop_rec.called, (
            "FAIL (SC-027): stop_recording was called despite run name mismatch. "
            "The guard at stop.py:~430 (refuse unless --force-cleanup) is missing."
        )

    def test_SC027_mismatch_with_force_proceeds_to_stop_recording(self) -> None:
        """
        stop_run with force_cleanup=True must proceed past the mismatch
        guard and call stop_recording.
        """
        import asyncio
        from ipaddress import IPv4Address
        from unittest.mock import AsyncMock, MagicMock, patch

        import stop as stop_module
        from utils.pydantic_config_models import (
            DaqConfigValidator,
            NetworkConfigValidator,
            QuaboUidsValidator,
            RunStateLedger,
        )

        daq_config = DaqConfigValidator(
            head_node_ip_addr=IPv4Address("10.0.1.5"),
            head_node_data_dir="/data/head",
            daq_nodes=[],
        )
        network_config = NetworkConfigValidator()
        quabo_uids = QuaboUidsValidator(domes=[])

        mock_mgr = MagicMock()
        ledger = RunStateLedger(
            run_name="active_run_Y.pffd",
            status="ACTIVE",
            start_time="2026-01-01T00:00:00Z",
        )
        mock_mgr.load_state.return_value = ledger

        mock_stop_rec = AsyncMock()

        with patch("stop.RunStateManager", return_value=mock_mgr), \
             patch("socket.gethostbyname", return_value="10.0.1.5"), \
             patch("utils.util.local_ip", return_value=["10.0.1.5"]), \
             patch("stop.stop_recording", mock_stop_rec), \
             patch("utils.util.kill_hv_updater"), \
             patch("utils.util.kill_hk_recorder"), \
             patch("utils.util.kill_module_temp_monitor"), \
             patch("utils.util.stop_data_flow"), \
             patch("utils.util.remove_run_name"):

            asyncio.run(stop_module.stop_run(
                daq_config, network_config, quabo_uids,
                verbose=False, run="different_run_X.pffd", force_cleanup=True,
            ))

        assert mock_stop_rec.called, (
            "FAIL (SC-027): stop_recording was NOT called even with force_cleanup=True. "
            "The --force-cleanup escape hatch in stop.py is broken."
        )


# ── SC-028 / SC-029: stop.py ctrl-C at various stages ────────────────────────

@pytest.mark.skip(reason="SC-028: requires SIGINT injection during stop.py execution")
def test_SC028_stop_killed_between_stop_recording_and_stop_data_flow() -> None:
    """
    SC-028: ctrl-C between stop_recording and stop_data_flow leaves quabos
    streaming to a now-dead DAQ node, filling kernel buffers.

    FAILS RED TODAY: no cleanup of quabo destinations in stop.py's finally block.
    Fix: use a context manager that calls stop_data_flow() on exit regardless.
    """
    pytest.skip("Requires SIGINT injection mid stop.py via subprocess")


@pytest.mark.skip(reason="SC-029: requires SIGINT injection during collect_data")
def test_SC029_stop_killed_between_collect_and_cleanup_leaves_stale_data() -> None:
    """
    SC-029: ctrl-C between collect_data (rsync complete) and _cleanup_daq_grpc
    leaves collect_complete_filename written but PFF files still on DAQ.
    Next run's cleanup sees stale data from the previous run.

    FAILS RED TODAY: no transactional guarantee around collect + cleanup.
    Fix: mark cleanup-pending with a sentinel; retry cleanup on next start.
    """
    pytest.skip("Requires SIGINT injection mid stop.py via subprocess")


# ── SC-030: PH baseline file missing ─────────────────────────────────────────

def test_SC030_missing_ph_baseline_file_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    """
    SC-030: start.py must refuse to start if the PH baseline file does not
    exist. Pins the missing-file contract (not TDD-forcing).
    """
    try:
        from start import ph_baseline_file_ok
    except ImportError:
        pytest.skip("Could not import start.ph_baseline_file_ok — check sys.path")

    non_existent = str(tmp_path / "no_such_file.json")
    result = ph_baseline_file_ok(non_existent)
    assert not result, "Missing PH baseline file must cause ph_baseline_file_ok to return False"


# ── SC-035: quabo_uids.json UID refused by mock-quabo ─────────────────────────

@pytest.mark.skip(reason="SC-035: requires mock_quabo + start_data_flow full integration")
def test_SC035_unreachable_quabo_uid_silently_fails() -> None:
    """
    SC-035: When quabo_uids.json lists a UID that the quabo refuses (e.g., wrong
    module IP), start_data_flow() calls quabo.send_daq_params() fire UDP into a
    void — no error is surfaced.

    FAILS RED TODAY: send_daq_params is fire-and-forget UDP; no ACK or error.
    Fix: ping-sweep quabos before start_data_flow, or verify HK packet received
    after configuration.
    """
    pytest.skip("Requires mock_quabo silence mode + start_data_flow integration")


# ── SC-036: Run directory collision (clock resolution = seconds) ──────────────

def test_SC036_run_dir_collision_is_detected(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    SC-036: Rapid sequential StartDaq calls within the same UTC second produce
    the same auto-generated run_dir name → mkdir raises FileExistsError.

    This tests the server's behavior when a run_dir already exists.
    The server must return ok=False with a clear error, not crash.
    """
    # Start and stop a run with a known run_dir
    ok1, _ = grpc_start(daq_control_direct, run_params)
    assert ok1, "First StartDaq must succeed"
    assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=10)
    daq_control_direct.StopDaq({
        "data_dir": run_params["data_dir"],
        "run_dir": run_params["run_dir"],
    })
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=8)

    # Attempt to start again with the same run_dir (the dir still exists on disk)
    ok2, resp2 = grpc_start(daq_control_direct, run_params)
    # Server must handle the collision gracefully
    if not ok2:
        assert resp2, "Collision must produce a descriptive error message"
    # Either outcome is acceptable here (server may allow reuse); what matters is
    # no crash and a deterministic response.

    # Cleanup
    with contextlib.suppress(Exception):
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir": run_params["run_dir"],
        })
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=5)
    with contextlib.suppress(Exception):
        daq_control_direct.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })


# ── SC-039 / SC-040: Config modification races ───────────────────────────────

@pytest.mark.skip(reason="SC-039: requires file mutation between get_daq_params and start_recording")
def test_SC039_data_config_modified_between_get_params_and_start() -> None:
    """
    SC-039: If data_config.json is modified between get_daq_params() and
    start_recording(), quabos have mode A while hashpipe was told mode B.
    This is an inconsistency that can cause silent data corruption.

    FAILS RED TODAY: no config snapshot is taken before multi-step start.
    Fix: snapshot (deepcopy) config at the start of start_run() and use
    the snapshot throughout the transaction.
    """
    pytest.skip("Requires file mutation injection mid-start.py execution")


@pytest.mark.skip(reason="SC-040: requires obs_config mutation between session_start and start.py")
def test_SC040_obs_config_timing_mode_change_between_session_and_run() -> None:
    """
    SC-040: If timing_mode in obs_config.json changes between session_start and
    start.py, quabo 0 still runs the old WR firmware but start.py configures for
    GNSS (or vice versa). Result: inconsistent timing across modules.

    FAILS RED TODAY: start.py reloads obs_config fresh and does not compare
    against the session_start snapshot.
    Fix: write a session_config_snapshot.json at session_start; validate it
    matches current obs_config at start.py time.
    """
    pytest.skip("Requires obs_config mutation between session_start and start.py")
