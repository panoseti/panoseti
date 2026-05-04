"""
core_happy_path — Parameterized observing-loop test.

Runs the full pseti observing cycle — start → record → stop → transfer queue
→ verify — for each data-config variant in DATA_CONFIGS.  No power cycle is
required between variants; maroc_config + mask_config are re-applied by the
active_data_config fixture.

Required state: PH_CALIBRATED (ensured by the booted_calibrated session fixture).
Leaves state: PH_CALIBRATED (no hardware reconfiguration after each run).

Class: happy_path (batch_priority = 0)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from ci.hardware_software.core_happy_path.checks import (
    daq,
    hk,
    ledger as ledger_checks,
    queue as queue_checks,
)
from ci.hardware_software.core import ledger as ledger_core
from control.pseti import app
from control.utils import config_file

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.hw_class("happy_path"),
    pytest.mark.timeout(600),
]

_RUN_DURATION_S = 30
_TRANSFER_TIMEOUT_S = 180


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _invoke(runner, args: list[str]) -> str:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (
        f"pseti {' '.join(args)} failed (exit {result.exit_code}):\n{result.output}"
    )
    return result.output


# ---------------------------------------------------------------------------
# Happy path test
# ---------------------------------------------------------------------------

def test_happy_path(booted_calibrated, active_data_config, runner, topology) -> None:
    """
    Full observing loop:
      1. Isolate the transfer queue (stop daemon so queue assertions are deterministic).
      2. Start a 30-second --no-hv run.
      3. In-flight checks: ledger ACTIVE, HK in Redis, DAQ disk growing.
      4. Stop the run.
      5. Assert the run is enqueued in the transfer pending bucket.
      6. Start the transfer daemon and wait for completion.
      7. Assert the run is in the completed bucket only.
      8. Verify the manifest.
    """
    daq_cfg = config_file.get_daq_config()
    first_node = daq_cfg.daq_nodes[0]
    quabo_addrs = topology.quabo_ips()

    # Step 1: isolate queue (stop daemon — idempotent if already stopped)
    runner.invoke(app, ["xfr", "stop"])  # not asserted — ok if already stopped
    time.sleep(2)

    # Step 2: start run
    logger.info("[HAPPY-PATH] data_config=%s: pseti start --no-hv --nsecs %d", active_data_config, _RUN_DURATION_S)
    _invoke(runner, ["start", "-y", "--no-hv", "--nsecs", str(_RUN_DURATION_S)])
    run_name = ledger_core.current_run_name()
    logger.info("[HAPPY-PATH] run_name=%s", run_name)

    # Step 3a: ledger should be ACTIVE immediately
    ledger_checks.is_active(run_name)

    # Step 3b: HK packets should appear in Redis within 30s
    boardlocs = [a.boardloc for a in quabo_addrs]
    hk.redis_populated(boardlocs, timeout=30)

    # Step 3c: DAQ node should be writing data
    daq.disk_growing(
        host=str(first_node.ip_addr),
        port=50051,
        run_name=run_name,
        min_bytes=500_000,
        window_s=10,
    )

    # Step 4: wait for the run duration to elapse, then stop
    # pseti start --nsecs auto-stops; give it a buffer
    ledger_checks.reaches("RECORDING_ENDED", timeout=_RUN_DURATION_S + 60)

    # Step 5: run should be enqueued in transfer pending
    queue_checks.only_in_bucket(run_name, "pending")

    # Step 6: start transfer daemon and wait for completion
    _invoke(runner, ["xfr", "start"])
    queue_checks.only_in_bucket(run_name, "completed", timeout=_TRANSFER_TIMEOUT_S)

    # Step 7: verify manifest
    _invoke(runner, ["xfr", "verify", run_name])

    logger.info("[HAPPY-PATH] data_config=%s run=%s: ALL CHECKS PASSED", active_data_config, run_name)
