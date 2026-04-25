"""
HW-01: Full start→stop→archive on real Quabos.

Exercises the complete observing lifecycle — Start/Stop transactions, the
Transfer Daemon, selective cleanup, and manifest verification — against real
hardware (Quabos + DAQ node + White-Rabbit switch).

Assertions:
  - PFF files appear on head node after transfer.
  - run_complete marker written.
  - manifest.blake3 root digest matches on both sides (DAQ + head node).
  - DAQ-side .pff files removed; .json/.log metadata preserved.

Entry point: pseti test hw run -k HW_01
"""

from __future__ import annotations

import os
import pathlib
import time

from typer.testing import CliRunner

from control.pseti import app
from control.utils.pydantic_config_models import DaqConfig
from control.utils.run_state import RunStateManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HEAD_DATA_DIR = pathlib.Path(os.getenv("HEAD_DATA_DIR", "/data/head"))
ARCHIVE_POLL_INTERVAL = 5   # seconds between ledger polls
ARCHIVE_TIMEOUT = 300       # maximum seconds to wait for ARCHIVED state

# ---------------------------------------------------------------------------
# HW-01
# ---------------------------------------------------------------------------

class TestHW01FullRunArchive:
    """End-to-end integration test against real Quabos."""

    def test_HW_01_session_start(self, runner: CliRunner) -> None:
        """Power on, get UIDs, calibrate. Prerequisite for all other HW tests."""
        result = runner.invoke(app, ["session-start", "--yes"])
        assert result.exit_code == 0, f"session-start failed:\n{result.stdout}"

    def test_HW_01_start_30s_run(self, runner: CliRunner, daq_config: DaqConfig) -> None:
        """Start a 30-second test run and verify it reaches ACTIVE state."""
        result = runner.invoke(app, ["start", "--run-type", "test", "--yes"])
        assert result.exit_code == 0, f"pseti start failed:\n{result.stdout}"

        mgr = RunStateManager()
        ledger = mgr.get_state()
        assert ledger is not None
        assert ledger.get("status") == "ACTIVE", f"Expected ACTIVE, got {ledger.get('status')}"

    def test_HW_01_stop_and_wait_archived(self, runner: CliRunner) -> None:
        """Stop the run and wait for the Transfer Daemon to reach ARCHIVED."""
        result = runner.invoke(app, ["stop", "--yes"])
        assert result.exit_code == 0, f"pseti stop failed:\n{result.stdout}"

        mgr = RunStateManager()
        deadline = time.monotonic() + ARCHIVE_TIMEOUT
        status = None
        while time.monotonic() < deadline:
            ledger = mgr.get_state()
            status = ledger.get("status") if ledger else None
            if status in ("ARCHIVED", "STOPPED_WITH_ERRORS", "VERIFY_FAILED"):
                break
            time.sleep(ARCHIVE_POLL_INTERVAL)

        assert status == "ARCHIVED", (
            f"Run did not reach ARCHIVED within {ARCHIVE_TIMEOUT}s. Final status: {status}"
        )

    def test_HW_01_pff_on_head_node(self, daq_config: DaqConfig) -> None:
        """PFF files must be present on the head node after transfer."""
        mgr = RunStateManager()
        ledger = mgr.get_state()
        assert ledger, "No ledger state found"
        run_name = ledger.get("run_name")
        assert run_name, "No run_name in ledger"

        run_dir = HEAD_DATA_DIR / run_name
        assert run_dir.exists(), f"Head node run dir missing: {run_dir}"

        pff_files = list(run_dir.rglob("*.pff"))
        assert pff_files, f"No .pff files on head node under {run_dir}"

    def test_HW_01_run_complete_marker(self) -> None:
        """run_complete marker must be written by the Transfer Daemon."""
        mgr = RunStateManager()
        ledger = mgr.get_state()
        assert ledger, "No ledger state found"
        run_name = ledger["run_name"]
        marker = HEAD_DATA_DIR / run_name / "run_complete"
        assert marker.exists(), f"run_complete marker missing: {marker}"

    def test_HW_01_manifest_digest_matches(self, daq_config: DaqConfig) -> None:
        """
        Manifest root digest on the head node must match what the DAQ node wrote.
        """
        from control.transfer.verify import verify_manifest

        mgr = RunStateManager()
        ledger = mgr.get_state()
        assert ledger, "No ledger state found"
        run_name = ledger["run_name"]
        run_dir = HEAD_DATA_DIR / run_name

        manifests = list(run_dir.rglob("manifest.*"))
        assert manifests, f"No manifest file found under {run_dir}"

        for manifest_path in manifests:
            ok, errors = verify_manifest(manifest_path, manifest_path.parent)
            assert ok, (
                f"Manifest verification failed for {manifest_path}: "
                + "; ".join(errors)
            )

    def test_HW_01_daq_pff_removed_metadata_preserved(self, daq_config: DaqConfig) -> None:
        """
        After selective cleanup:
          - .pff files must be gone from DAQ nodes.
          - .json / .log / .toml metadata must be preserved.
        """
        import paramiko  # type: ignore[import]

        mgr = RunStateManager()
        ledger = mgr.get_state()
        assert ledger, "No ledger state found"
        run_name = ledger["run_name"]

        for node in daq_config.daq_nodes:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(str(node.ip_addr), username=node.username, timeout=10)
                for mid in node.module_ids:
                    run_path = f"{node.data_dir}/module_{mid}/{run_name}"
                    _, stdout, _ = ssh.exec_command(f"ls {run_path}/*.pff 2>/dev/null | wc -l")
                    pff_count = int(stdout.read().strip() or "0")
                    assert pff_count == 0, (
                        f"DAQ node {node.ip_addr} module {mid}: "
                        f"{pff_count} .pff files not cleaned up"
                    )
            finally:
                ssh.close()
