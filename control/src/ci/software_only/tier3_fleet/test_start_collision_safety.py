"""
tier3_fleet/test_start_collision_safety.py — Run collision and rollback safety.

Verifies:
1. A second start attempt is rejected if a run is ACTIVE.
2. The active run is NOT disturbed by the aborted second attempt.
3. --force-reset archives the first run and starts a new one cleanly.

These tests use no_data=True so no Hashpipe or Docker is required.
The ledger still reaches ACTIVE; only the hardware-touching paths are skipped.

Ported from software_only/tier3_fleet/test_start_collision_safety.py.
"""

from __future__ import annotations

import unittest.mock
import uuid
from contextlib import ExitStack

import pytest

from ci.software_only.infra.spec import FleetSpec
from ci.software_only.infra.workspace import Workspace
from control.start import start_run
from control.utils.pydantic_config_models import RunStatus
from control.utils.run_state import RunStateManager

pytestmark = pytest.mark.tier3


def _patched_start() -> ExitStack:
    """Context manager that patches all hardware-touching start_run paths."""
    stack = ExitStack()
    for target, retval in [
        ("control.start.ph_baseline_file_ok", True),
        ("control.start._check_daq_reachability", None),
        ("control.start._check_quabo_reachability", None),
        ("control.start.start_data_flow", None),
        ("control.start.make_run_dirs", None),
        ("control.start.util.start_hk_recorder", None),
        ("control.start.util.write_run_name", None),
        ("control.start.config_file.associate", None),
        ("control.start.config_file.show_daq_assignments", None),
    ]:
        if retval is not None:
            stack.enter_context(unittest.mock.patch(target, return_value=retval))
        else:
            stack.enter_context(unittest.mock.patch(target))
    return stack


@pytest.mark.parametrize("pseti_workspace", [FleetSpec.minimal_unit()], indirect=True)
@pytest.mark.asyncio
async def test_when_run_active_then_second_start_blocked(
    pseti_workspace: Workspace,
) -> None:
    """An aborted second start must NOT disturb the pre-existing ACTIVE run.

    With no_data=True the ledger reaches ACTIVE without requiring Hashpipe.
    The second attempt sees the ACTIVE ledger and returns None.
    The first run's ledger entry must survive unchanged.
    """
    daq_cfg = pseti_workspace.topology.daq.model_copy(  # type: ignore[union-attr]
        update={"head_node_data_dir": str(pseti_workspace.root / "head_data")}
    )
    obs_cfg = pseti_workspace.topology.obs
    quabo_uids = pseti_workspace.topology.quabo_uids
    data_cfg = pseti_workspace.topology.data
    net_cfg = pseti_workspace.topology.network

    run1_name = f"run1_{uuid.uuid4().hex[:8]}.pffd"

    with _patched_start():
        res1 = await start_run(
            obs_cfg,  # type: ignore[arg-type]
            daq_cfg,
            quabo_uids,  # type: ignore[arg-type]
            data_cfg,  # type: ignore[arg-type]
            net_cfg,  # type: ignore[arg-type]
            no_hv=True,
            no_redis=True,
            no_data=True,
            run_name=run1_name,
            no_check_daq=True,
        )

    assert res1 == run1_name, f"First start_run should succeed, got: {res1}"
    mgr = RunStateManager()
    ledger = mgr.load_state()
    assert ledger is not None
    assert ledger.status == RunStatus.ACTIVE

    # Attempt Run 2 — must be blocked by the ACTIVE ledger.
    run2_name = f"run2_{uuid.uuid4().hex[:8]}.pffd"
    with _patched_start():
        res2 = await start_run(
            obs_cfg,  # type: ignore[arg-type]
            daq_cfg,
            quabo_uids,  # type: ignore[arg-type]
            data_cfg,  # type: ignore[arg-type]
            net_cfg,  # type: ignore[arg-type]
            no_hv=True,
            no_redis=True,
            no_data=True,
            run_name=run2_name,
            no_check_daq=True,
        )

    assert res2 is None, "Second start_run should return None (blocked by ACTIVE ledger)"

    # Run 1 must be undisturbed.
    ledger = mgr.load_state()
    assert ledger is not None
    assert ledger.run_name == run1_name
    assert ledger.status == RunStatus.ACTIVE


@pytest.mark.parametrize("pseti_workspace", [FleetSpec.minimal_unit()], indirect=True)
@pytest.mark.asyncio
async def test_when_force_reset_then_first_run_archived_and_new_starts(
    pseti_workspace: Workspace,
) -> None:
    """force_reset=True archives the first ACTIVE run and starts the second.

    The first start completes successfully; the second start with force_reset=True
    archives the first ledger entry and proceeds to ACTIVE with the new run name.
    """
    daq_cfg = pseti_workspace.topology.daq.model_copy(  # type: ignore[union-attr]
        update={"head_node_data_dir": str(pseti_workspace.root / "head_data")}
    )
    obs_cfg = pseti_workspace.topology.obs
    quabo_uids = pseti_workspace.topology.quabo_uids
    data_cfg = pseti_workspace.topology.data
    net_cfg = pseti_workspace.topology.network

    run1_name = f"run1_{uuid.uuid4().hex[:8]}.pffd"

    with _patched_start():
        await start_run(
            obs_cfg,  # type: ignore[arg-type]
            daq_cfg,
            quabo_uids,  # type: ignore[arg-type]
            data_cfg,  # type: ignore[arg-type]
            net_cfg,  # type: ignore[arg-type]
            no_hv=True,
            no_redis=True,
            no_data=True,
            run_name=run1_name,
            no_check_daq=True,
        )

    ledger = RunStateManager().load_state()
    assert ledger is not None and ledger.status == RunStatus.ACTIVE

    run2_name = f"run2_{uuid.uuid4().hex[:8]}.pffd"
    with _patched_start():
        res2 = await start_run(
            obs_cfg,  # type: ignore[arg-type]
            daq_cfg,
            quabo_uids,  # type: ignore[arg-type]
            data_cfg,  # type: ignore[arg-type]
            net_cfg,  # type: ignore[arg-type]
            no_hv=True,
            no_redis=True,
            no_data=True,
            run_name=run2_name,
            no_check_daq=True,
            force_reset=True,
        )

    assert res2 == run2_name, f"start_run with force_reset should succeed: got {res2}"

    ledger = RunStateManager().load_state()
    assert ledger is not None
    assert ledger.run_name == run2_name
    assert ledger.status == RunStatus.ACTIVE
