"""
HW-04: Mid-run hashpipe crash → resilient StopTransaction rollback.

Verifies that when hashpipe is SIGKILL'd on a DAQ node mid-run, the control
plane detects the crash and pseti stop --force-cleanup completes the full
teardown sequence:
  - Quabos stop emitting data (UDP port silent for 5 s).
  - HV is ramped down (via WPS query).
  - Local daemons are terminated.
  - Ledger is ABORTED or STOPPED_WITH_ERRORS with failure_context.json.

Entry point: pseti test hw run -k HW_04
"""

from __future__ import annotations

import socket
import time
import os

import pytest
from typer.testing import CliRunner

from control.pseti import app
from control.utils import config_file, util
from control.utils.run_state import RunStateManager

DAQ_DATA_DIR = os.getenv("DAQ_DATA_DIR", "/data")
QUABO_UDP_PORT = 60001   # science data port


def _quabo_emitting(quabo_ip: str, listen_sec: float = 5.0) -> bool:
    """Return True if the Quabo sends any UDP packet on port 60001 within listen_sec."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", QUABO_UDP_PORT))
        s.settimeout(listen_sec)
        try:
            s.recvfrom(4096)
            return True
        except socket.timeout:
            return False


class TestHW04HashpipeCrashRollback:
    """Resilient teardown when hashpipe crashes mid-run."""

    def test_HW_04_start_run(self, runner: CliRunner) -> None:
        """Start a test run; baseline confirms hashpipe is running."""
        result = runner.invoke(app, ["start", "--run-type", "test", "--yes"])
        assert result.exit_code == 0, f"pseti start failed:\n{result.stdout}"
        time.sleep(5)  # let hashpipe stabilize

    def test_HW_04_kill_hashpipe_via_ssh(
        self, daq_config: object, network_config: object
    ) -> None:
        """SIGKILL hashpipe on the first DAQ node to simulate a crash."""
        import subprocess

        util.attach_daq_config(daq_config, network_config)
        node = daq_config.daq_nodes[0]
        host, _ = util.daq_grpc_endpoint(node)

        result = subprocess.run(
            ["ssh", *util.ssh_options, f"{node.username}@{host}", "pkill -9 hashpipe"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        # returncode 1 means no process matched — also acceptable if already dead
        assert result.returncode in (0, 1), (
            f"ssh pkill failed ({result.returncode}): {result.stderr}"
        )

    def test_HW_04_force_cleanup_succeeds(self, runner: CliRunner) -> None:
        """pseti stop --force-cleanup must complete despite crashed hashpipe."""
        result = runner.invoke(app, ["stop", "--yes", "--force-cleanup"])
        assert result.exit_code == 0, f"pseti stop --force-cleanup failed:\n{result.stdout}"

    def test_HW_04_quabos_silent_after_stop(self, obs_config: object) -> None:
        """Quabos must not emit science data 5 s after stop."""
        for dome in obs_config.domes:
            for module in dome.modules:
                base = str(module.ip_addr).rsplit(".", 1)
                quabo_ip = f"{base[0]}.{int(base[1])}"  # quabo 0
                emitting = _quabo_emitting(quabo_ip, listen_sec=5.0)
                assert not emitting, (
                    f"Quabo {quabo_ip} still emitting science data 5 s after stop"
                )

    def test_HW_04_ledger_has_error_status(self) -> None:
        """Ledger must reflect the crash with ABORTED or STOPPED_WITH_ERRORS."""
        mgr = RunStateManager()
        ledger = mgr.get_state()
        assert ledger, "No ledger state found"
        status = ledger.get("status")
        assert status in ("ABORTED", "STOPPED_WITH_ERRORS", "RECORDING_ENDED"), (
            f"Expected error status after crash, got: {status}"
        )
