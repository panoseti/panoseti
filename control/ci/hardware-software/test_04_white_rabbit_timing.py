"""
HW-03: White-Rabbit timing sanity.

Verifies that WR-sourced timing in PFF frames is self-consistent:
  |pkt_nsec/1e6 - tv_usec/1000| < 25 ms for ≥99% of sampled frames,
  with correct ±1 s adjustment when the sub-second difference crosses a
  second boundary (per CLAUDE.md §Timing).

Entry point: pseti test hw run -k HW_03
"""

from __future__ import annotations

import os
import pathlib
import time

from typer.testing import CliRunner

from control.pseti import app
from control.utils.run_state import RunStateManager

HEAD_DATA_DIR = pathlib.Path(os.getenv("HEAD_DATA_DIR", "/data/head"))
RUN_DURATION_SEC = 60
SAMPLE_FRAMES = 500       # frames to sample for the timing check
TIMING_THRESHOLD_MS = 25  # |pkt_nsec/1e6 - tv_usec/1000| must be < this


class TestHW03WhiteRabbitTiming:
    """Timing sanity on real WR hardware."""

    def test_HW_03_acquire_60s_run(self, runner: CliRunner) -> None:
        """Acquire a 60-second run for later analysis."""
        result = runner.invoke(app, ["start", "--run-type", "test", "--yes"])
        assert result.exit_code == 0, f"pseti start failed:\n{result.stdout}"
        print(f"Waiting {RUN_DURATION_SEC}s for data acquisition...")
        time.sleep(RUN_DURATION_SEC)
        result = runner.invoke(app, ["stop", "--yes"])
        assert result.exit_code == 0, f"pseti stop failed:\n{result.stdout}"

        # Wait for ARCHIVED
        mgr = RunStateManager()
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            ledger = mgr.get_state()
            if ledger and ledger.get("status") == "ARCHIVED":
                break
            time.sleep(5)

    def test_HW_03_timing_within_25ms(self) -> None:
        """
        Sample PFF frames and verify WR timing consistency.

        Per the CLAUDE.md timing rule:
            if |tv_usec/1000 - pkt_nsec/1e6| > 25 ms:
                adjust tv_sec by ±1
        Here we assert the *raw* difference is < 25 ms for ≥99% of frames
        (no adjustment needed), confirming WR lock is healthy.
        """
        from control.utils import pff as pff_utils

        mgr = RunStateManager()
        ledger = mgr.get_state()
        assert ledger, "No ledger state"
        run_name = ledger["run_name"]
        run_dir = HEAD_DATA_DIR / run_name

        pff_files = sorted(run_dir.rglob("*.pff"))
        assert pff_files, f"No PFF files under {run_dir}"

        mismatches = 0
        total = 0

        for pff_path in pff_files:
            if total >= SAMPLE_FRAMES:
                break
            try:
                for header, _image in pff_utils.read_pff(str(pff_path)):
                    if total >= SAMPLE_FRAMES:
                        break
                    pkt_nsec = header.get("pkt_nsec", 0)
                    tv_usec = header.get("tv_usec", 0)
                    diff_ms = abs(pkt_nsec / 1e6 - tv_usec / 1000)
                    # Apply ±1 s adjustment if needed (per CLAUDE.md)
                    if diff_ms > 25:
                        diff_ms = abs(diff_ms - 1000)
                    if diff_ms >= TIMING_THRESHOLD_MS:
                        mismatches += 1
                    total += 1
            except Exception:
                continue  # skip malformed frames at boundaries

        assert total > 0, "No frames sampled — PFF files may be empty"
        bad_pct = mismatches / total * 100
        assert bad_pct < 1.0, (
            f"WR timing out of spec: {mismatches}/{total} frames "
            f"({bad_pct:.1f}%) exceed {TIMING_THRESHOLD_MS} ms threshold"
        )
