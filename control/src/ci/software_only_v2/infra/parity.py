"""
infra/parity.py — v1↔v2 equivalence harness used during the sunset period.

Purpose
-------
During the phased v1→v2 migration, every ported test scenario is registered
here via @parity_test.  The registration proves that:
  1. The scenario exercises the same observable behavior in both suites.
  2. A sunset reviewer can grep the registry to see what remains unported.

Sunset gate: v1 cannot be deleted until every test in
``software_only/tier{1..5}_*/`` has a passing parity entry, plus a 7-day
soak with zero v2-only failures.

Usage
-----
Registering a parity scenario::

    from ci.software_only_v2.infra.parity import parity_test, run_scenario

    @parity_test(
        v1_fixtures=("mock_workspace", "session_fleet"),
        v2_fixtures=("pseti_workspace", "session_fleet"),
        scenario="two_node_start_stop",
    )
    def two_node_start_stop_assertions(probe, run_name: str) -> None:
        assert probe.ledger_status() == "ARCHIVED"
        assert probe.any_pff_files(run_name, head=True)

Running assertions from a v2 test::

    def test_two_node_start_stop(pseti_workspace, session_fleet):
        ...do the scenario work...
        run_scenario("two_node_start_stop", probe=pseti_workspace.state_probe, run_name=...)

Generating a coverage report::

    from ci.software_only_v2.infra.parity import parity_coverage_report
    print(parity_coverage_report())
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Internal registry
# ---------------------------------------------------------------------------

@dataclass
class ParityScenario:
    """Metadata for one registered parity scenario."""

    scenario: str
    v1_fixtures: tuple[str, ...]
    v2_fixtures: tuple[str, ...]
    assertions_fn: Callable[..., None]
    source_file: str = ""
    source_line: int = 0
    description: str = ""


_REGISTRY: dict[str, ParityScenario] = {}


# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------

def parity_test(
    *,
    v1_fixtures: tuple[str, ...],
    v2_fixtures: tuple[str, ...],
    scenario: str,
    description: str = "",
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Register an assertions function as a parity scenario.

    The decorated function is returned unchanged so it can also be used as a
    plain Python callable by v2 test bodies.

    Args:
        v1_fixtures: Names of the v1 pytest fixtures needed to supply the
            probe environment (for documentation + coverage tracking only;
            they are not injected by this decorator).
        v2_fixtures: Names of the v2 pytest fixtures for the same scenario.
        scenario: Unique stable identifier for this scenario (snake_case).
        description: One-line summary (defaults to the function's docstring
            first line).
    """
    def decorator(fn: Callable[..., None]) -> Callable[..., None]:
        _desc = description or (inspect.getdoc(fn) or "").splitlines()[0]
        frame = inspect.stack()[1]
        _REGISTRY[scenario] = ParityScenario(
            scenario=scenario,
            v1_fixtures=tuple(v1_fixtures),
            v2_fixtures=tuple(v2_fixtures),
            assertions_fn=fn,
            source_file=frame.filename,
            source_line=frame.lineno,
            description=_desc,
        )
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def run_scenario(scenario: str, **kwargs: Any) -> None:
    """Execute the registered assertions function for *scenario*.

    Call this from a v2 (or v1) test body after setting up the observable
    state.  Any kwargs are forwarded to the assertions function.

    Raises:
        KeyError: if *scenario* is not registered.
        AssertionError: propagated from the assertions function on failure.
    """
    entry = _REGISTRY[scenario]
    entry.assertions_fn(**kwargs)


def get_scenario(scenario: str) -> ParityScenario:
    """Return the registered ParityScenario for *scenario*.

    Raises:
        KeyError: if *scenario* is not registered.
    """
    return _REGISTRY[scenario]


# ---------------------------------------------------------------------------
# Coverage & reporting
# ---------------------------------------------------------------------------

def all_scenarios() -> dict[str, ParityScenario]:
    """Return a snapshot of all registered parity scenarios."""
    return dict(_REGISTRY)


def parity_coverage_report(*, v1_test_ids: list[str] | None = None) -> str:
    """Generate a human-readable coverage report.

    Args:
        v1_test_ids: Optional list of v1 test node IDs to check against the
            registry.  If provided, uncovered v1 tests are listed.

    Returns:
        A formatted string suitable for printing or logging.
    """
    lines: list[str] = ["=== v1↔v2 Parity Coverage ===", ""]

    if not _REGISTRY:
        lines.append("  (no scenarios registered)")
        return "\n".join(lines)

    for name, entry in sorted(_REGISTRY.items()):
        v1 = ", ".join(entry.v1_fixtures)
        v2 = ", ".join(entry.v2_fixtures)
        src = f"{entry.source_file}:{entry.source_line}" if entry.source_file else "unknown"
        lines.append(f"  ✓ {name}")
        if entry.description:
            lines.append(f"      {entry.description}")
        lines.append(f"      v1: ({v1})")
        lines.append(f"      v2: ({v2})")
        lines.append(f"      registered at: {src}")
        lines.append("")

    lines.append(f"Total: {len(_REGISTRY)} scenario(s) registered.")

    if v1_test_ids:
        # Simple heuristic: check if each v1 test module base name appears in
        # any registered scenario name or description.
        registered_text = " ".join(
            f"{name} {entry.description}"
            for name, entry in _REGISTRY.items()
        ).lower()
        uncovered = []
        for tid in v1_test_ids:
            # tid like "software_only/tier2_logic/test_ledger.py::TestLock::test_acquire"
            base = tid.split("::")[-1].lower().replace("test_", "")
            if base not in registered_text:
                uncovered.append(tid)

        if uncovered:
            lines.append("")
            lines.append(f"⚠  {len(uncovered)} v1 test(s) without parity coverage:")
            for tid in uncovered:
                lines.append(f"   - {tid}")
        else:
            lines.append("✓  All provided v1 test IDs have parity coverage.")

    return "\n".join(lines)


def sunset_gate_met(required_scenarios: list[str]) -> tuple[bool, list[str]]:
    """Check whether all required scenarios have been registered.

    Returns:
        (met, missing) where met is True iff all required scenarios are
        registered, and missing is the list of any that are not.
    """
    missing = [s for s in required_scenarios if s not in _REGISTRY]
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Built-in parity scenarios (registered alongside their v2 equivalents)
# ---------------------------------------------------------------------------

@parity_test(
    v1_fixtures=("mock_workspace", "pseti_workspace"),
    v2_fixtures=("pseti_workspace",),
    scenario="workspace_seven_config_files",
    description="Workspace materializes all 7 config files and sets PSETI_* env vars",
)
def _workspace_seven_config_files(config_dir: Any, expected_files: list[str]) -> None:
    """Assert all 7 config JSON files exist in config_dir."""
    import pathlib
    cdir = pathlib.Path(config_dir)
    for fname in expected_files:
        assert (cdir / fname).exists(), f"{fname} missing from config_dir"


@parity_test(
    v1_fixtures=("mock_workspace",),
    v2_fixtures=("pseti_workspace",),
    scenario="ledger_starts_empty",
    description="Fresh workspace has no active ledger entry",
)
def _ledger_starts_empty(state_dir: Any) -> None:
    """Assert RunStateManager reports no active run."""
    import pathlib
    from control.utils.run_state import RunStateManager
    mgr = RunStateManager(base_dir=pathlib.Path(state_dir))
    assert mgr.load_state() is None


@parity_test(
    v1_fixtures=("mock_workspace", "session_fleet"),
    v2_fixtures=("pseti_workspace", "session_fleet"),
    scenario="two_node_start_stop",
    description="Two-node fleet starts, records, stops, and ledger reaches RECORDING_ENDED",
)
def _two_node_start_stop(probe: Any, expected_status: str = "RECORDING_ENDED") -> None:
    """Assert ledger reaches expected_status after start+stop cycle."""
    actual = probe.ledger_status()
    assert actual == expected_status, (
        f"Ledger status: expected {expected_status!r}, got {actual!r}"
    )


@parity_test(
    v1_fixtures=("mock_workspace",),
    v2_fixtures=("pseti_workspace",),
    scenario="config_validator_passes",
    description="GlobalConfigValidator approves the materialized topology",
)
def _config_validator_passes(topology: Any) -> None:
    """Assert GlobalConfigValidator returns True for the given topology."""
    import copy
    from control.utils.global_validator import GlobalConfigValidator

    obs = topology.obs
    data = topology.data
    daq = copy.deepcopy(topology.daq)
    network = topology.network
    uids = copy.deepcopy(topology.quabo_uids)

    validator = GlobalConfigValidator({
        "obs": obs, "data": data, "daq": daq,
        "network": network, "firmware": None, "uids": uids,
    })
    assert validator.validate_all_rules(), (
        "GlobalConfigValidator failed on topology — unexpected validation error"
    )


@parity_test(
    v1_fixtures=("mock_workspace", "session_fleet"),
    v2_fixtures=("pseti_workspace", "session_fleet"),
    scenario="fleet_boot_and_healthy",
    description="All DAQ nodes in a fleet boot and pass gRPC health checks",
)
def _fleet_boot_and_healthy(fleet: Any) -> None:
    """Assert every DAQ node in the fleet is alive and gRPC-healthy."""
    for i in range(len(fleet.daq_nodes)):
        client = fleet.daq_control_client(i)
        resp = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": False})
        assert resp is not None, f"DAQ node {i} health check returned None"


@parity_test(
    v1_fixtures=("mock_workspace", "session_fleet"),
    v2_fixtures=("pseti_workspace", "session_fleet"),
    scenario="grpc_status_returns_idle",
    description="StatusDaq RPC returns idle (hashpipe_pid==0) on a freshly booted node",
)
def _grpc_status_returns_idle(fleet: Any, node_index: int = 0) -> None:
    """Assert freshly-booted DAQ node reports no running hashpipe process."""
    client = fleet.daq_control_client(node_index)
    resp = client.StatusDaq({"data_dir": "/data", "check_hashpipe_running": True})
    assert resp.hashpipe_pid == 0, (
        f"Expected hashpipe_pid=0 on idle node, got {resp.hashpipe_pid}"
    )


@parity_test(
    v1_fixtures=("mock_workspace", "chaos_fleet"),
    v2_fixtures=("pseti_workspace", "chaos_fleet"),
    scenario="grpc_inject_unavailable",
    description="gRPC proxy fault injection causes UnavailableError on client calls",
)
def _grpc_inject_unavailable(fleet: Any, node_index: int = 0) -> None:
    """Assert that a gRPC call raised UnavailableError during the fault window.

    The caller is responsible for setting up the fault; this function checks
    the effect attribute set on the fleet object by the fault context manager.
    """
    from panoseti_grpc.grpc_utils.exceptions import UnavailableError
    last_exc = getattr(fleet, "_last_grpc_exc", None)
    assert isinstance(last_exc, UnavailableError), (
        f"Expected UnavailableError during fault window, got: {last_exc!r}"
    )


@parity_test(
    v1_fixtures=("mock_workspace", "chaos_fleet"),
    v2_fixtures=("pseti_workspace", "chaos_fleet"),
    scenario="process_kill_and_restart",
    description="Killing the gRPC server process is detected and the process restarts",
)
def _process_kill_and_restart(fleet: Any, node_index: int = 0, timeout: int = 30) -> None:
    """Assert the gRPC server on the given node is alive after a kill+restart cycle."""
    alive = fleet.chaos.proc.wait_alive(
        fleet.daq_nodes[node_index], "pseti-grpc", timeout=timeout
    )
    assert alive, (
        f"gRPC server on node {node_index} did not restart within {timeout}s"
    )


@parity_test(
    v1_fixtures=("mock_workspace", "session_fleet"),
    v2_fixtures=("pseti_workspace", "session_fleet"),
    scenario="data_collection_happy_path",
    description="Start→record→stop cycle completes and ledger reaches RECORDING_ENDED",
)
def _data_collection_happy_path(probe: Any, expected_status: str = "RECORDING_ENDED") -> None:
    """Assert ledger reaches expected_status after a full start+stop cycle."""
    actual = probe.ledger_status()
    assert actual == expected_status, (
        f"Ledger status after happy-path run: expected {expected_status!r}, got {actual!r}"
    )


@parity_test(
    v1_fixtures=("mock_workspace", "session_fleet"),
    v2_fixtures=("pseti_workspace", "session_fleet"),
    scenario="cleanup_blocked_while_hashpipe_running",
    description="CleanupData RPC is rejected while hashpipe is still running",
)
def _cleanup_blocked_while_hashpipe_running(fleet: Any, node_index: int = 0) -> None:
    """Assert that CleanupData raises an error while hashpipe is active.

    The caller must have started hashpipe before invoking this assertion.
    The assertion checks the fleet's recorded last RPC error.
    """
    from panoseti_grpc.grpc_utils.exceptions import FailedPreconditionError
    last_exc = getattr(fleet, "_last_cleanup_exc", None)
    assert isinstance(last_exc, FailedPreconditionError), (
        f"Expected FailedPreconditionError when cleaning up active run, got: {last_exc!r}"
    )


@parity_test(
    v1_fixtures=("mock_workspace", "session_fleet"),
    v2_fixtures=("pseti_workspace_session", "session_fleet"),
    scenario="two_node_independent_lifecycle",
    description="Independent nodes in a fleet can be started and stopped without interference",
)
def _two_node_independent_lifecycle(node_0_running: bool, node_1_running: bool) -> None:
    """Assert the running state of two nodes matches expected independence."""
    assert node_0_running is False, "Node 0 should have been stopped"
    assert node_1_running is True, "Node 1 should still be running"


@parity_test(
    v1_fixtures=("mock_workspace", "daq_control_gateway"),
    v2_fixtures=("pseti_workspace_session", "session_fleet"),
    scenario="gateway_consistency",
    description="Gateway client sees the same server state as a direct client",
)
def _gateway_consistency(direct_running: bool, gateway_running: bool) -> None:
    """Assert that direct and gateway views of the server state are identical."""
    assert direct_running == gateway_running, (
        f"State mismatch: direct={direct_running}, gateway={gateway_running}"
    )


@parity_test(
    v1_fixtures=("mock_workspace", "mock_rsync_transfer"),
    v2_fixtures=("pseti_workspace", "mock_rsync_transfer"),
    scenario="transfer_partial_recovery",
    description="Transfer queue recovers from partial rsync failure and succeeds on retry",
)
def _transfer_partial_recovery(success: bool, archived: bool) -> None:
    """Assert that the transfer eventually succeeded and the run is ARCHIVED."""
    assert success is True, "Transfer failed to recover"
    assert archived is True, "Run status did not reach ARCHIVED"


@parity_test(
    v1_fixtures=("mock_workspace", "transfer_job_factory"),
    v2_fixtures=("pseti_workspace", "transfer_job_factory"),
    scenario="transfer_selective_cleanup",
    description="Transfer cleanup deletes .pff files but preserves metadata",
)
def _transfer_selective_cleanup(pff_count: int, meta_exists: bool) -> None:
    """Assert that .pff files were removed but meta.json remains."""
    assert pff_count == 0, f"Expected 0 .pff files, found {pff_count}"
    assert meta_exists is True, "Metadata file was accidentally deleted"
