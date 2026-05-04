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

# pseti start (no --nsecs) returns immediately after the ledger reaches ACTIVE,
# leaving the run going in background daemons. We sleep this long before stopping.
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


def _stop_if_active(runner) -> None:
    """Stop the run if the ledger is currently ACTIVE (teardown helper)."""
    try:
        current = ledger_core.load()
        if current and str(current.status) in ("ACTIVE", "STARTING"):
            logger.warning("[HAPPY-PATH] teardown: stopping active run")
            runner.invoke(app, ["stop", "-y"])
    except Exception as exc:
        logger.warning("[HAPPY-PATH] teardown: stop attempt failed: %s", exc)


# ---------------------------------------------------------------------------
# Happy path test
# ---------------------------------------------------------------------------

def test_happy_path(booted_calibrated, active_data_config, runner, topology) -> None:
    """
    Full observing loop:
      1. Isolate the transfer queue (stop daemon so queue assertions are deterministic).
      2. Start run with no --nsecs: pseti start returns as soon as ledger reaches ACTIVE.
      3. In-flight checks: ledger ACTIVE, HK in Redis, DAQ disk growing.
      4. Sleep _RUN_DURATION_S to accumulate data, then stop explicitly.
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

    # Step 2: start run — returns immediately after ledger reaches ACTIVE.
    # (pseti start --nsecs blocks the full duration; without it, start returns
    # after setup so in-flight checks can run in the main thread.)
    logger.info("[HAPPY-PATH] data_config=%s: pseti start --no-hv", active_data_config)
    _invoke(runner, ["start", "-y", "--no-hv"])
    run_name = ledger_core.current_run_name()
    logger.info("[HAPPY-PATH] run_name=%s", run_name)

    try:
        # Step 3a: ledger must be ACTIVE immediately after start returns
        ledger_checks.is_active(run_name)

        # Step 3b: HK packets should appear in Redis within 30s
        boardlocs = [a.boardloc for a in quabo_addrs]
        hk.redis_populated(boardlocs, timeout=30)

        # Step 3c: DAQ node should be writing data
        daq.disk_growing(
            node=first_node,
            daq_config=daq_cfg,
            run_name=run_name,
            min_bytes=500_000,
            window_s=10,
        )

        # Step 4: let it record for a while, then stop explicitly
        logger.info("[HAPPY-PATH] sleeping %ds then stopping", _RUN_DURATION_S)
        time.sleep(_RUN_DURATION_S)
        _invoke(runner, ["stop", "-y"])

    except Exception:
        _stop_if_active(runner)
        raise

    ledger_checks.reaches("RECORDING_ENDED", timeout=60)

    # Step 5: run should be enqueued in transfer pending
    queue_checks.only_in_bucket(run_name, "pending")

    # Step 6: start transfer daemon and wait for completion
    _invoke(runner, ["xfr", "start"])
    queue_checks.only_in_bucket(run_name, "completed", timeout=_TRANSFER_TIMEOUT_S)

    # Step 7: verify manifest
    _invoke(runner, ["xfr", "verify", run_name])

    logger.info("[HAPPY-PATH] data_config=%s run=%s: ALL CHECKS PASSED", active_data_config, run_name)
