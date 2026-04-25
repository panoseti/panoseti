"""
HW-07: Transfer pipeline end-to-end on real hardware.

Exercises the complete stop → TransferJob enqueue → Transfer Daemon →
ARCHIVED lifecycle against real Quabos and a real DAQ node.

Assertions:
  - After pseti stop, ledger status is RECORDING_ENDED (fast-path).
  - A pending TransferJob exists and port_forwarding round-trips correctly.
  - Transfer Daemon drives the run to ARCHIVED within 10 minutes.
  - run_complete marker is written on the head node.
  - All manifest.{blake3,xxh3_128,sha256} files pass verify_manifest().
  - DAQ-node .pff files are removed; .json/.log/.toml metadata preserved.

Entry point: pseti test hw run -k HW_07
"""

from __future__ import annotations

import os
import pathlib
import time
import tomllib

import pytest
from typer.testing import CliRunner

from control.pseti import app
from control.transfer.models import TransferJob
from control.transfer.queue import TransferQueue
from control.transfer.verify import verify_manifest
from control.utils import config_file, util
from control.utils.run_state import RunStateManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEAD_DATA_DIR = pathlib.Path(os.getenv("HEAD_DATA_DIR", "/data/head"))
ARCHIVE_POLL_INTERVAL = 5    # seconds between ledger polls
ARCHIVE_TIMEOUT = 600        # 10 minutes maximum to reach ARCHIVED
TERMINAL_STATUSES = {"ARCHIVED", "STOPPED_WITH_ERRORS", "VERIFY_FAILED", "TRANSFER_FAILED"}


# ---------------------------------------------------------------------------
# HW-07 test class
# ---------------------------------------------------------------------------

class TestHW07TransferPipeline:
    """End-to-end transfer pipeline test against real hardware."""

    # ── Phase 1: start a short run ──────────────────────────────────────────

    def test_HW_07_start_run(self, runner: CliRunner) -> None:
        """Start a 30-second test run and verify ACTIVE state."""
        result = runner.invoke(app, ["start", "--run-type", "hwsw_transfer", "--yes"])
        assert result.exit_code == 0, f"pseti start failed:\n{result.stdout}"
        mgr = RunStateManager()
        ledger = mgr.load_state()
        assert ledger is not None, "Ledger must exist after start"
        assert ledger.status == "ACTIVE", (
            f"Expected ACTIVE, got {ledger.status!r}"
        )

    # ── Phase 2: stop and verify fast-path ──────────────────────────────────

    def test_HW_07_stop_fast_path(self, runner: CliRunner) -> None:
        """pseti stop must complete quickly and leave ledger in RECORDING_ENDED."""
        t0 = time.monotonic()
        result = runner.invoke(app, ["stop", "--yes"])
        elapsed = time.monotonic() - t0
        assert result.exit_code == 0, f"pseti stop failed:\n{result.stdout}"
        assert elapsed < 60.0, (
            f"pseti stop took {elapsed:.1f}s — expected <60s for the fast path"
        )
        mgr = RunStateManager()
        ledger = mgr.load_state()
        assert ledger is not None
        assert ledger.status == "RECORDING_ENDED", (
            f"Expected RECORDING_ENDED after stop, got {ledger.status!r}"
        )

    # ── Phase 2b: verify TransferJob was enqueued with port_forwarding ──────

    def test_HW_07_transfer_job_enqueued(self) -> None:
        """A pending TransferJob must exist and port_forwarding must round-trip."""
        daq_config = config_file.get_daq_config()
        network_config = config_file.get_network_config()
        util.attach_daq_config(daq_config, network_config)

        mgr = RunStateManager()
        ledger = mgr.load_state()
        assert ledger is not None
        run_name = ledger.run_name

        tq = TransferQueue()
        pending = tq.list_jobs("pending")
        active = tq.list_jobs("active")
        assert run_name in (pending + active), (
            f"Expected {run_name!r} in pending or active queue.\n"
            f"pending={pending}, active={active}"
        )

        # Locate the job TOML (may have moved to active/ already)
        for bucket in ("pending", "active"):
            job_path = (
                tq._queue / bucket / f"{run_name}.job.toml"
            )
            if job_path.exists():
                with open(job_path, "rb") as f:
                    data = tomllib.load(f)
                job = TransferJob.model_validate(data)
                # Regression: port_forwarding must round-trip
                for job_node, cfg_node in zip(job.daq_nodes, daq_config.daq_nodes):
                    has_pf = cfg_node.port_forwarding is not None and cfg_node.port_forwarding.status
                    if has_pf:
                        assert job_node.port_forwarding is not None, (
                            f"port_forwarding was dropped for node {cfg_node.ip_addr}"
                        )
                        assert str(job_node.port_forwarding.gw_ip) == str(cfg_node.port_forwarding.gw_ip)
                        assert job_node.port_forwarding.port == cfg_node.port_forwarding.port
                    else:
                        assert job_node.port_forwarding is None or not job_node.port_forwarding.status
                break

    # ── Phase 3: wait for ARCHIVED ──────────────────────────────────────────

    def test_HW_07_wait_for_archived(self) -> None:
        """Transfer Daemon must drive the run to ARCHIVED within 10 minutes."""
        mgr = RunStateManager()
        deadline = time.monotonic() + ARCHIVE_TIMEOUT
        status = None
        while time.monotonic() < deadline:
            ledger = mgr.load_state()
            status = ledger.status if ledger else None
            if status in TERMINAL_STATUSES:
                break
            time.sleep(ARCHIVE_POLL_INTERVAL)

        assert status == "ARCHIVED", (
            f"Run did not reach ARCHIVED within {ARCHIVE_TIMEOUT}s. "
            f"Final status: {status!r}"
        )

    # ── Phase 4: integrity proofs ────────────────────────────────────────────

    def test_HW_07_run_complete_marker(self) -> None:
        """run_complete marker must be written by the Transfer Daemon."""
        mgr = RunStateManager()
        ledger = mgr.load_state()
        assert ledger is not None
        marker = HEAD_DATA_DIR / ledger.run_name / "run_complete"
        assert marker.exists(), f"run_complete missing: {marker}"

    def test_HW_07_manifest_verification(self) -> None:
        """All head-node manifests must pass verify_manifest() after transfer."""
        mgr = RunStateManager()
        ledger = mgr.load_state()
        assert ledger is not None
        run_dir = HEAD_DATA_DIR / ledger.run_name
        assert run_dir.exists(), f"Head run dir missing: {run_dir}"

        found_any = False
        for algo in ("blake3", "xxh3_128", "sha256"):
            mf = run_dir / f"manifest.{algo}"
            if not mf.exists():
                continue
            found_any = True
            ok, errs = verify_manifest(mf, run_dir)
            assert ok, (
                f"manifest.{algo} verification failed:\n" + "\n".join(errs)
            )
        assert found_any, f"No manifest files found under {run_dir}"

    # ── Phase 5: selective cleanup proof ────────────────────────────────────

    def test_HW_07_daq_pff_removed_metadata_preserved(self) -> None:
        """After selective cleanup: .pff gone from DAQ nodes; .json/.log/.toml preserved."""
        import paramiko  # type: ignore[import]

        daq_config = config_file.get_daq_config()
        network_config = config_file.get_network_config()
        util.attach_daq_config(daq_config, network_config)

        mgr = RunStateManager()
        ledger = mgr.load_state()
        assert ledger is not None
        run_name = ledger.run_name

        for node in daq_config.daq_nodes:
            if not node.module_ids:
                continue
            pf = node.port_forwarding
            use_pf = pf is not None and pf.status
            ssh_host = str(pf.gw_ip) if use_pf else str(node.ip_addr)
            ssh_port = pf.port if use_pf else 22

            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(ssh_host, port=ssh_port, username=node.username, timeout=15)
                for mid in node.module_ids:
                    run_path = f"{node.data_dir}/module_{mid}/{run_name}"
                    # .pff must be gone
                    _, out, _ = ssh.exec_command(
                        f"ls {run_path}/*.pff 2>/dev/null | wc -l"
                    )
                    pff_count = int(out.read().strip() or "0")
                    assert pff_count == 0, (
                        f"Node {node.ip_addr} module {mid}: "
                        f"{pff_count} .pff files still present after cleanup"
                    )
                    # metadata (.json/.log/.toml) must be preserved
                    _, out2, _ = ssh.exec_command(
                        f"ls {run_path}/*.json {run_path}/*.log {run_path}/*.toml 2>/dev/null | wc -l"
                    )
                    meta_count = int(out2.read().strip() or "0")
                    assert meta_count > 0, (
                        f"Node {node.ip_addr} module {mid}: "
                        f"no metadata (.json/.log/.toml) found — selective cleanup may have been too aggressive"
                    )
            finally:
                ssh.close()
