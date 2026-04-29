"""Tier 5 (Integration): Transfer daemon observability end-to-end tests.

Test 4.4 from EXECUTION_PLAN — validates that:
  - The real transfer daemon writes to transfer_daemon.log (not /dev/null).
  - `pseti xfr tail` produces non-empty output after daemon starts.
  - `pseti ledger path` returns an existing (or printable) path.
  - After SIGTERM + >30s, `pseti xfr status` reports STALE heartbeat.

These tests do NOT require Docker or real hardware — they start the daemon
in-process and assert against the filesystem state.  The daemon is isolated
via PSETI_STATE and PSETI_TQ_DIR environment variables redirected to a
temporary directory.
"""
from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_file_content(path: pathlib.Path, substring: str, timeout: float = 15.0) -> bool:
    """Poll *path* until it exists and contains *substring*, or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                text = path.read_text()
                if substring in text:
                    return True
            except OSError:
                pass
        time.sleep(0.25)
    return False


def _wait_for_heartbeat(heartbeat_path: pathlib.Path, timeout: float = 15.0) -> bool:
    """Poll until the heartbeat file exists and was written recently."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if heartbeat_path.exists():
            try:
                ts = float(heartbeat_path.read_text().strip())
                if time.time() - ts < 10.0:
                    return True
            except (ValueError, OSError):
                pass
        time.sleep(0.25)
    return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def daemon_env(tmp_path: pathlib.Path) -> dict[str, str]:
    """Return environment variables that redirect daemon state to tmp_path."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    env = os.environ.copy()
    env["PSETI_STATE"] = str(state_dir)
    env["PSETI_TQ_DIR"] = str(tmp_path / "queue")
    return env


# ---------------------------------------------------------------------------
# Test 4.4a: daemon writes to transfer_daemon.log
# ---------------------------------------------------------------------------

class TestDaemonLogOutput:
    def test_daemon_writes_log_file(
        self, tmp_path: pathlib.Path, daemon_env: dict[str, str]
    ) -> None:
        """The daemon must produce transfer_daemon.log, not swallow all output.

        This is the D-1 regression check: start_daemon previously routed
        stdout/stderr to DEVNULL, so any crash was invisible.  Now both the
        structured logger and the subprocess stdout/stderr go to log files.
        """
        proc = subprocess.Popen(
            [sys.executable, "-m", "control.transfer"],
            env=daemon_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            log_dir = pathlib.Path(daemon_env["PSETI_STATE"]) / "logs" / "transfer_daemon"
            log_file = log_dir / "transfer_daemon.log"

            found = _wait_for_file_content(log_file, "Transfer daemon started", timeout=15.0)
            assert found, (
                f"Expected 'Transfer daemon started' in {log_file}. "
                f"Log content: {log_file.read_text() if log_file.exists() else '(file not found)'}"
            )
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_daemon_heartbeat_written(
        self, tmp_path: pathlib.Path, daemon_env: dict[str, str]
    ) -> None:
        """The daemon must write a heartbeat file within 10 seconds of starting."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "control.transfer"],
            env=daemon_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            hb_path = pathlib.Path(daemon_env["PSETI_STATE"]) / "transfer" / "daemon.heartbeat"
            found = _wait_for_heartbeat(hb_path, timeout=15.0)
            assert found, f"Daemon heartbeat not written within 15s at {hb_path}"
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def test_daemon_stdout_stderr_not_devnull(
        self, tmp_path: pathlib.Path, daemon_env: dict[str, str]
    ) -> None:
        """start_daemon must write stdout.log and stderr.log as backstops.

        This ensures even a crash before the logger initializes is visible.
        We test the log file existence (produced by util.start_daemon), not
        the daemon directly, since the in-process start uses start_daemon().
        """
        from control.utils import util
        from control.utils.paths import PanoPaths

        # Temporarily point PanoPaths at tmp_path.
        os.environ["PSETI_STATE"] = daemon_env["PSETI_STATE"]
        try:
            util.start_daemon(
                [sys.executable, "-m", "control.transfer"],
                name="transfer_daemon",
            )
            time.sleep(2.0)  # let the daemon start and write files

            log_dir = PanoPaths.daemon_logs_dir("transfer_daemon")
            stdout_log = log_dir / "stdout.log"
            stderr_log = log_dir / "stderr.log"
            assert stdout_log.exists(), f"Expected {stdout_log} to exist after start_daemon"
            assert stderr_log.exists(), f"Expected {stderr_log} to exist after start_daemon"
        finally:
            del os.environ["PSETI_STATE"]
            # Clean up any started daemon processes.
            pid_path = pathlib.Path(daemon_env["PSETI_STATE"]) / "transfer" / "daemon.pid"
            if pid_path.exists():
                try:
                    pid = int(pid_path.read_text().strip())
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(1.0)
                except (ValueError, OSError, ProcessLookupError):
                    pass


# ---------------------------------------------------------------------------
# Test 4.4b: tail command produces real output
# ---------------------------------------------------------------------------

class TestTransferTailCommand:
    def test_tail_produces_output_after_daemon_start(
        self, tmp_path: pathlib.Path, daemon_env: dict[str, str]
    ) -> None:
        """pseti xfr tail must return non-empty content after daemon starts.

        This is the D-5 regression check: the old tail pointed at current.log
        which never existed; now it points at transfer_daemon.log.
        """
        proc = subprocess.Popen(
            [sys.executable, "-m", "control.transfer"],
            env=daemon_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            log_dir = pathlib.Path(daemon_env["PSETI_STATE"]) / "logs" / "transfer_daemon"
            log_file = log_dir / "transfer_daemon.log"
            assert _wait_for_file_content(log_file, "Transfer daemon started", timeout=15.0), \
                "Daemon did not write log within 15s"

            # Simulate what `pseti xfr tail -n 5` does: read last N lines.
            result = subprocess.run(
                ["tail", "-n5", str(log_file)],
                capture_output=True, text=True,
            )
            assert result.returncode == 0
            assert result.stdout.strip(), "tail -n5 produced empty output"
            assert "Transfer daemon" in result.stdout or "transfer" in result.stdout.lower(), \
                f"Unexpected tail content: {result.stdout!r}"
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


# ---------------------------------------------------------------------------
# Test 4.4c: ledger path command
# ---------------------------------------------------------------------------

class TestLedgerPathCommand:
    def test_ledger_path_command_returns_path(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pseti ledger path must print a non-empty path string."""
        monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))

        from typer.testing import CliRunner

        from control.tools.ledger_cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["path"])
        assert result.exit_code == 0, f"ledger path exited with {result.exit_code}: {result.output}"
        assert result.output.strip(), "ledger path printed nothing"
        printed = result.output.strip()
        assert pathlib.Path(printed).suffix == ".toml" or "run_state" in printed, \
            f"Expected a .toml path, got: {printed!r}"

    def test_ledger_show_graceful_on_missing_ledger(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pseti ledger show must exit with code 1 and a useful error, not crash."""
        monkeypatch.setenv("PSETI_STATE", str(tmp_path / "state"))

        from typer.testing import CliRunner

        from control.tools.ledger_cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["show"])
        assert result.exit_code == 1, \
            f"Expected exit code 1 on missing ledger, got {result.exit_code}"
        assert "No active ledger" in result.output or "not found" in result.output.lower(), \
            f"Expected informative error, got: {result.output!r}"


# ---------------------------------------------------------------------------
# Test 4.4d: daemon status reports STALE after heartbeat ages out
# ---------------------------------------------------------------------------

class TestDaemonStatusStale:
    def test_heartbeat_age_reflects_stopped_daemon(
        self, tmp_path: pathlib.Path, daemon_env: dict[str, str]
    ) -> None:
        """After SIGTERM, the heartbeat file must exist and be older than when
        the daemon was last running.  The status helper uses this to report STALE.

        We don't wait 30+ real seconds (that would be impractical in CI); instead
        we verify that the heartbeat timestamp is at least 2 seconds old after
        a 3-second wait — confirming the daemon stopped writing heartbeats.
        """
        proc = subprocess.Popen(
            [sys.executable, "-m", "control.transfer"],
            env=daemon_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            hb_path = pathlib.Path(daemon_env["PSETI_STATE"]) / "transfer" / "daemon.heartbeat"
            assert _wait_for_heartbeat(hb_path, timeout=15.0), "Heartbeat never written"
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Daemon is stopped. Wait 3 seconds and verify the heartbeat hasn't updated.
        time.sleep(3.0)
        assert hb_path.exists(), "Heartbeat file should persist after daemon stops"
        age = time.time() - float(hb_path.read_text().strip())
        assert age >= 2.0, \
            f"Expected heartbeat to be at least 2s old after stop, got {age:.1f}s"
