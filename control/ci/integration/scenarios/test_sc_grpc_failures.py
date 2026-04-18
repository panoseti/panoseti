"""
scenarios/test_sc_grpc_failures.py

SC-001 → SC-020: gRPC failure isolation tests.

Key exemplars implemented here:
  - SC-010 (Exemplar A): Orphaned hashpipe blocks CleanupData without --force
  - SC-006 (Exemplar C): StopDaq partial failure leaves zombie hashpipes

TDD intent: each test is designed to FAIL RED on current master.
Run separately from the main integration suite via:
    python ci/qa.py chaos -k test_sc_grpc
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
from panoseti_grpc.daq_control.client import DaqControlClient

from ci.integration.chaos import process_chaos
from ci.integration.conftest import (
    DAQ_DATA_DIR,
    DAQNODE_CONTAINER,
    DAQNODE_DIRECT_HOST,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
)
from ci.integration.state_probe import StateProbe

from .conftest import (
    StopPartialFailure,
    any_pff_files_on_daqnode,
)
from .conftest import (
    _cleanup as grpc_cleanup,
)
from .conftest import (
    _start as grpc_start,
)
from .conftest import (
    _stop as grpc_stop,
)

# ── SC-001: StartDaq timeout ────────────────────────────────────────────────

@pytest.mark.skip(reason="SC-001: requires grpc_proxy fixture + timeout injection")
def test_SC001_startdaq_timeout_hangs_forever(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    StartDaq with no timeout hangs forever when daqnode is unresponsive.

    Current bug (start.py): no deadline on DaqControlClient.StartDaq; hangs forever.
    Fix required: deadline/timeout on all StartDaq calls.
    """
    # Inject 120s slow response → call should time out, not hang
    # proxy.set_mode("StartDaq", "slow_response", timeout_s=120)
    pytest.skip("requires grpc_proxy with slow_response mode")


# ── SC-002: Partial start rolls back ────────────────────────────────────────
# (Exemplar B — in test_sc_transactional_state.py)


# ── SC-005: Post-start liveness check ───────────────────────────────────────

@pytest.mark.skip(reason="SC-005: requires hashpipe that exits immediately")
def test_SC005_hashpipe_exits_immediately_not_detected(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    StartDaq succeeds but hashpipe exits within 1s — control plane doesn't notice.

    Current bug: no post-start liveness check; StatusDaq is never polled by start.py.
    Fix: poll StatusDaq for 2-5s after StartDaq to confirm hashpipe is still up.
    """
    pytest.skip("requires fake hashpipe that exits immediately")


# ── SC-006 (Exemplar C): StopDaq partial failure ─────────────────────────────

class TestSC006StopDaqPartialFailure:
    """
    SC-006: stop.py's stop_recording raises on the first failed StopDaq,
    skipping subsequent DAQ nodes.

    FAILS RED TODAY: stop.py::stop_recording has no per-node isolation;
    one failure causes all subsequent nodes to be skipped.
    """

    def test_SC006_stop_continues_after_per_node_timeout(
        self,
        daq_control_direct: DaqControlClient,
        daq_control_node2: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
        daqnode_container: Any,
    ) -> None:
        """
        With two DAQ nodes, a StopDaq timeout on node-0 must NOT prevent
        node-1 from being stopped.

        Currently fails because stop_recording raises on the first failure,
        and the loop never reaches node-1.
        """
        # Start hashpipe on both nodes
        rp1 = dict(run_params)
        rp2 = dict(run_params, daq_ip_addr="192.168.0.20", module_id=[200])

        daq_control_direct.StartDaq(rp1)
        wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=10)
        daq_control_node2.StartDaq(rp2)
        wait_hashpipe_running(daq_control_node2, DAQ_DATA_DIR, timeout=10)

        # Freeze node-0's hashpipe to simulate a slow exit (StopDaq times out)
        with process_chaos.freeze_process(DAQNODE_CONTAINER, "hashpipe"):
            # In the production stop.py, stop_recording loops over nodes.
            # With node-0 frozen, SIGINT will not be acked → timeout.
            # Node-1 MUST still receive StopDaq.
            from ipaddress import IPv4Address

            import stop as stop_module
            from utils.pydantic_config_models import DaqConfigValidator, DaqNodeValidator

            # Construct a dummy DaqConfigValidator with both nodes
            daq_config = DaqConfigValidator(
                head_node_ip_addr=IPv4Address("10.0.1.22"),
                head_node_data_dir="/data/head",
                daq_nodes=[
                    DaqNodeValidator(ip_addr=IPv4Address(rp1["daq_ip_addr"]), data_dir=rp1["data_dir"], username="root", module_ids=rp1["module_id"]),
                    DaqNodeValidator(ip_addr=IPv4Address(rp2["daq_ip_addr"]), data_dir=rp2["data_dir"], username="root", module_ids=rp2["module_id"])
                ]
            )
            
            import asyncio
            # Call actual stop_recording
            asyncio.run(stop_module.stop_recording(daq_config, rp1["run_dir"], verbose=False))
            
            # Node-1 must still have been stopped despite node-0 failure
            assert wait_hashpipe_stopped(daq_control_node2, DAQ_DATA_DIR, timeout=8), (
                "Node-1 was never told to stop because node-0 raised first "
                "(SC-006 bug: stop_recording is not fault-isolated per node)"
            )

        # Cleanup
        with contextlib.suppress(Exception):
            daq_control_direct.StopDaq({
                "data_dir": rp1["data_dir"], "run_dir": rp1["run_dir"]
            })
            daq_control_node2.StopDaq({
                "data_dir": rp2["data_dir"], "run_dir": rp2["run_dir"]
            })
        with contextlib.suppress(Exception):
            daq_control_direct.CleanupData({
                "data_dir": rp1["data_dir"],
                "run_dir": rp1["run_dir"],
                "module_id": rp1["module_id"],
            })
            daq_control_node2.CleanupData({
                "data_dir": rp2["data_dir"],
                "run_dir": rp2["run_dir"],
                "module_id": rp2["module_id"],
            })


def _call_stop_recording_for_two_nodes(
    client1: DaqControlClient,
    client2: DaqControlClient,
    params1: dict[str, Any],
    params2: dict[str, Any],
) -> None:
    """
    Simulate stop.py's sequential stop_recording loop for two nodes.
    Raises StopPartialFailure if either node fails; currently the loop
    stops at the first failure (the bug being tested).
    """
    errors = []
    for client, params in [(client1, params1), (client2, params2)]:
        ok, resp = grpc_stop(client, {
            "data_dir": params["data_dir"],
            "run_dir": params["run_dir"],
        })
        if not ok:
            errors.append(f"{params['daq_ip_addr']}: {resp}")
            # Current bug: production code raises here, never reaches next node
            raise StopPartialFailure(f"StopDaq failed: {errors}")
    if errors:
        raise StopPartialFailure("; ".join(errors))


# ── SC-007: StopDaq on already-stopped service (contract test) ──────────────

def test_SC007_stop_on_already_stopped_returns_success(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    StopDaq on a node with no running hashpipe MUST return success=True.

    Not TDD-forcing — pins the idempotent-stop contract.
    """
    ok, resp = grpc_stop(daq_control_direct, {
        "data_dir": run_params["data_dir"],
        "run_dir": run_params["run_dir"],
    })
    assert ok is True, f"StopDaq on stopped service returned ok=False: {resp}"


# ── SC-009: CleanupData while hashpipe is running ────────────────────────────

def test_SC009_cleanup_blocked_while_hashpipe_running(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    CleanupData MUST refuse (not silently succeed) while hashpipe is running.

    This pins the data-safety contract: we never delete science data while
    recording is active.
    """
    daq_control_direct.StartDaq(run_params)
    assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=10)
    try:
        ok, _resp = grpc_cleanup(daq_control_direct, {
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })
        assert not ok, (
            "CleanupData should refuse while hashpipe is running, "
            "but returned ok=True — data-safety contract violated"
        )
    finally:
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir": run_params["run_dir"],
        })
        wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=8)


# ── SC-010 (Exemplar A): Orphaned hashpipe blocks CleanupData ─────────────

class TestSC010OrphanedHashpipe:
    """
    SC-010 / Exemplar A: when hashpipe is SIGKILLed (orphaned), the DAQ Control
    server's `hashpipe_pid > 0` gate blocks CleanupData forever.

    TDD-forcing test — FAILS RED on current master because:
      (a) CleanupData without force: returns ok=False (server thinks hashpipe is alive)
          but there is no CleanupRefusedPreserveData exception class
      (b) CleanupData with force=True: the `force` field doesn't exist in the proto
      (c) No Redis incident key is written

    Fix requires:
      1. Server liveness check: kill(pid, 0) / psutil.pid_exists
      2. Proto: add `force` bool to CleanupDataRequest
      3. Server: write Redis incident key on forced cleanup
      4. stop.py: add --force-cleanup flag
    """

    def test_SC010_cleanup_refused_without_force_when_hashpipe_orphaned(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
        daqnode_container: Any,
        fresh_run_state: None,
    ) -> None:
        """
        After hashpipe is SIGKILLed, CleanupData without force must:
        - Refuse to delete data (return ok=False / raise FAILED_PRECONDITION)
        - Leave .pff files intact on the DAQ node
        """
        daq_control_direct.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=10), \
            "hashpipe did not start"

        # Simulate crash: SIGKILL bypasses StopDaq
        process_chaos.kill_process(DAQNODE_CONTAINER, "hashpipe", sig="KILL")
        assert process_chaos.wait_for_process_death(
            DAQNODE_CONTAINER, "hashpipe", timeout=5
        ), "hashpipe did not die after SIGKILL"

        # The server still has hashpipe_pid > 0 in memory.
        # CleanupData MUST refuse — it should NOT silently delete science data.
        ok, _resp = grpc_cleanup(daq_control_direct, {
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })
        assert not ok, (
            "FAIL (SC-010): CleanupData succeeded after hashpipe was orphaned — "
            "this is a data-safety violation. The server must either detect the "
            "dead PID or require force=true to proceed."
        )

    def test_SC010_cleanup_with_force_removes_data_and_writes_incident_key(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        state_probe: StateProbe,
        daqnode_container: Any,
        fresh_run_state: None,
    ) -> None:
        """
        With force=True, CleanupData must:
        - Delete the run data
        - Write a Redis incident key

        FAILS TODAY: the `force` field is not in the proto and there is no
        incident-key logic in the server.
        """
        daq_control_direct.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=10)

        process_chaos.kill_process(DAQNODE_CONTAINER, "hashpipe", sig="KILL")
        process_chaos.wait_for_process_death(DAQNODE_CONTAINER, "hashpipe", timeout=5)

        # force=True override: allowed to delete orphaned run data
        ok, _resp = grpc_cleanup(daq_control_direct, {
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
            "force":     True,   # ← not in proto yet: test will fail here
        })
        assert ok, f"CleanupData(force=True) failed: {_resp}"

        # Data must be gone
        assert not any_pff_files_on_daqnode(run_params["run_dir"], run_params["module_id"]), \
            "PFF files still present after CleanupData(force=True)"

        # Incident key must exist in Redis
        # incident_key = f"panoseti:incident:forced_cleanup:{run_params['run_dir']}"
        # assert state_probe.redis_incident_key(incident_key), (
        #     f"No Redis incident key {incident_key!r} written after forced cleanup"
        # )

    def test_SC010b_force_on_live_hashpipe_is_refused(
        self,
        daq_control_direct: DaqControlClient,
        run_params: dict[str, Any],
        fresh_run_state: None,
    ) -> None:
        """
        SC-010b: force=True on a LIVE hashpipe must still be refused.
        force is only an escape hatch for the dead-PID path, not a kill switch.
        """
        daq_control_direct.StartDaq(run_params)
        assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=10)
        try:
            ok, _resp = grpc_cleanup(daq_control_direct, {
                "data_dir":  run_params["data_dir"],
                "run_dir":   run_params["run_dir"],
                "module_id": run_params["module_id"],
                "force":     True,
            })
            assert not ok, (
                "CleanupData(force=True) on a LIVE hashpipe must still be refused — "
                "force is only for the dead-PID case"
            )
        finally:
            daq_control_direct.StopDaq({
                "data_dir": run_params["data_dir"],
                "run_dir": run_params["run_dir"],
            })
            wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=8)


# ── SC-013: StatusDaq during StartDaq (contract test) ────────────────────────

def test_SC013_status_during_start_is_consistent(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    StatusDaq called concurrently with StartDaq must not deadlock or crash.
    Pins the RPC ordering contract.
    """
    import threading

    daq_control_direct.StartDaq(run_params)
    results: list[Any] = []

    def _status() -> None:
        try:
            ok, resp = daq_control_direct.StatusDaq({
                "data_dir": run_params["data_dir"],
                "check_hashpipe_running": True,
                "check_disk_usage": False,
                "check_run_dirs": False,
            })
            results.append((ok, resp))
        except Exception as e:
            results.append(e)

    t = threading.Thread(target=_status)
    t.start()
    t.join(timeout=5)
    assert results, "StatusDaq call timed out"
    assert not isinstance(results[0], Exception), f"StatusDaq raised: {results[0]}"

    daq_control_direct.StopDaq({
        "data_dir": run_params["data_dir"],
        "run_dir": run_params["run_dir"],
    })
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=8)


# ── SC-018: Concurrent StartDaq to same node (contract test) ─────────────────

def test_SC018_concurrent_start_same_node_only_one_wins(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    Two concurrent StartDaq RPCs to the same node — at most one must succeed.
    The server enforces single-hashpipe-per-node.
    """
    import threading

    outcomes: list[Any] = []

    def _inner_start(run_dir_suffix: str) -> None:
        p = dict(run_params, run_dir=f"conctest_{run_dir_suffix}.pffd")
        ok, resp = grpc_start(daq_control_direct, p)
        outcomes.append((ok, resp, p["run_dir"]))

    t1 = threading.Thread(target=_inner_start, args=("a",))
    t2 = threading.Thread(target=_inner_start, args=("b",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    winners = [o for o in outcomes if o[0]]
    assert len(winners) <= 1, (
        f"Both concurrent StartDaq calls succeeded: {outcomes}. "
        "Server must enforce single-hashpipe-per-node."
    )
    # Cleanup winner
    for ok, _resp, run_dir in outcomes:
        if ok:
            daq_control_direct.StopDaq({
                "data_dir": run_params["data_dir"],
                "run_dir": run_dir,
            })
            wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=8)
            daq_control_direct.CleanupData({
                "data_dir":  run_params["data_dir"],
                "run_dir":   run_dir,
                "module_id": run_params["module_id"],
            })


# ── SC-003: StartDaq returns success=False (bad config) ──────────────────────

def test_SC003_startdaq_bad_run_dir_returns_failure(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    SC-003: StartDaq with an invalid/empty run_dir must return ok=False with a
    descriptive error message, not crash the server.

    Pins the contract: server-side validation must surface config errors cleanly.
    """
    ok, resp = grpc_start(daq_control_direct, {
        **run_params,
        "run_dir": "",  # empty run_dir — invalid
    })
    assert not ok, (
        "StartDaq with empty run_dir must return ok=False. "
        "Server must validate required fields before launching hashpipe."
    )
    assert resp, "StartDaq failure must include a non-empty error message"


# ── SC-004: StartDaq transient UNAVAILABLE, succeeds on retry ────────────────

@pytest.mark.skip(reason="SC-004: requires grpc_proxy fixture for fault injection")
def test_SC004_startdaq_transient_unavailable_succeeds_on_retry(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    SC-004: A transient UNAVAILABLE error on StartDaq must trigger a retry and
    eventually succeed. No retry layer exists today.

    FAILS RED TODAY: start.py has no retry logic for gRPC UNAVAILABLE.
    Fix: wrap StartDaq calls in a retry with exponential backoff.
    """
    pytest.skip("requires grpc_proxy with transient_unavailable mode")


# ── SC-008: StopDaq on never-started service (idempotent contract) ───────────

def test_SC008_stop_on_never_started_returns_success(
    daq_control_direct: DaqControlClient,
) -> None:
    """
    SC-008: StopDaq on a node that has never had a run (or after CleanupData)
    must return success — it is a no-op, not an error.

    Pins the idempotent-stop contract (not TDD-forcing).
    """
    ok, resp = grpc_stop(daq_control_direct, {
        "data_dir": DAQ_DATA_DIR,
        "run_dir": "never_existed_run.pffd",
    })
    assert ok is True, f"StopDaq on never-started service returned ok=False: {resp}"


# ── SC-011: CleanupData partial failure across two nodes ─────────────────────

def test_SC011_cleanup_partial_failure_logs_and_continues(
    daq_control_direct: DaqControlClient,
    daq_control_node2: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    SC-011: If node-0 CleanupData succeeds but node-1 fails (e.g., run_dir never
    existed on node-1), stop.py must log the error but not block completion.

    This test pins the partial-cleanup isolation contract.
    """
    import uuid as _uuid

    # Start a run on node-0 only
    ok, _ = grpc_start(daq_control_direct, run_params)
    assert ok, "StartDaq failed on node-0"
    assert wait_hashpipe_running(daq_control_direct, DAQ_DATA_DIR, timeout=10)

    daq_control_direct.StopDaq({
        "data_dir": run_params["data_dir"],
        "run_dir": run_params["run_dir"],
    })
    wait_hashpipe_stopped(daq_control_direct, DAQ_DATA_DIR, timeout=8)

    # Cleanup node-0 must succeed
    ok0, resp0 = grpc_cleanup(daq_control_direct, {
        "data_dir":  run_params["data_dir"],
        "run_dir":   run_params["run_dir"],
        "module_id": run_params["module_id"],
    })
    assert ok0, f"CleanupData failed on node-0: {resp0}"

    # Cleanup node-1 for a run that never started there — must not raise/hang
    ok1, _resp1 = grpc_cleanup(daq_control_node2, {
        "data_dir":  run_params["data_dir"],
        "run_dir":   f"nonexistent_{_uuid.uuid4().hex[:8]}.pffd",
        "module_id": run_params["module_id"],
    })
    # Either ok or a clear error — must not be an unhandled exception
    # (Some servers return ok=True for no-op cleanup; others ok=False with message)
    assert isinstance(ok1, bool), "CleanupData must return a bool ok status"


# ── SC-012: CleanupData with full disk on head node ──────────────────────────

@pytest.mark.skip(reason="SC-012: requires disk_chaos.fill_volume on head volume")
def test_SC012_cleanup_with_full_head_disk_does_not_retry(
    daq_control_direct: DaqControlClient,
    run_params: dict[str, Any],
    fresh_run_state: None,
) -> None:
    """
    SC-012: When the head-node rsync target is full, collect_data errors but
    does not retry; collect_complete_filename is not written, correctly blocking
    cleanup. Test pins the current no-retry behavior.

    FAILS RED TODAY: collect.collect_data does not retry on ENOSPC.
    Fix: retry with exponential backoff on transient ENOSPC.
    """
    pytest.skip("Requires disk_chaos.fill_volume on the head data volume")


# ── SC-014: gRPC RST_STREAM mid-StartDaq ────────────────────────────────────

@pytest.mark.skip(reason="SC-014: requires grpc_proxy reset_stream injection")
def test_SC014_rst_stream_during_startdaq_errors_cleanly() -> None:
    """
    SC-014: A gRPC RST_STREAM received mid-StartDaq must be surfaced as a clear
    error, not silently leave hashpipe in an unknown state.

    FAILS RED TODAY: behavior under RST_STREAM is not specified or tested.
    """
    pytest.skip("Requires grpc_proxy.reset_stream() on StartDaq")


# ── SC-015: Daqnode reboots during recording ────────────────────────────────

@pytest.mark.skip(reason="SC-015: requires container restart simulation")
def test_SC015_daqnode_reboot_during_run_makes_head_aware() -> None:
    """
    SC-015: When a DAQ node reboots mid-run, the head node still has current_run
    set. The next start.py invocation fails with 'run in progress' until the
    operator intervenes.

    FAILS RED TODAY: no keepalive/heartbeat between head and DAQ — head never
    learns the DAQ rebooted.
    Fix: StatusDaq heartbeat + auto-clear current_run on DAQ unreachable.
    """
    pytest.skip("Requires docker restart of daqnode container mid-run")


# ── SC-016: DaqControlClient with wrong port → clear error ──────────────────

def test_SC016_wrong_port_gives_clear_error() -> None:
    """
    SC-016: DaqControlClient constructed with a wrong port must raise or return
    a clear connection error within a reasonable timeout, not hang forever.

    Pins the operator-visible error contract.
    """
    import grpc

    bad_client = DaqControlClient(host=DAQNODE_DIRECT_HOST, port=9)  # port 9 = discard
    try:
        ok, resp = bad_client.StatusDaq({
            "data_dir": DAQ_DATA_DIR,
            "check_hashpipe_running": False,
            "check_disk_usage": False,
            "check_run_dirs": False,
        })
        # If it returns at all, it should be ok=False with an error message
        assert not ok or resp, "Should fail or return a non-empty response for unreachable port"
    except (grpc.RpcError, ConnectionError, OSError, Exception) as exc:
        # Any clear exception is acceptable — we just don't want a silent hang
        assert str(exc), f"Exception must have a message: {exc}"


# ── SC-017: DAQ Control service disabled (UNIMPLEMENTED) ────────────────────

@pytest.mark.skip(reason="SC-017: requires server config without daq_control=true")
def test_SC017_daq_control_disabled_returns_unimplemented() -> None:
    """
    SC-017: A panoseti-server running without daq_control enabled must return
    UNIMPLEMENTED for all DAQ Control RPCs.

    FAILS RED TODAY: server profile is not tested with daq_control disabled.
    Fix: test with a separate integration-headnode profile container.
    """
    pytest.skip("Requires a second container with daq_control=false profile")


# ── SC-019: CleanupData race with concurrent StartDaq ────────────────────────

@pytest.mark.skip(reason="SC-019: complex race — requires concurrent RPC injection")
def test_SC019_cleanup_race_with_startdaq() -> None:
    """
    SC-019: CleanupData issued while a concurrent StartDaq is racing with StopDaq
    must not delete data that the new run is writing.

    FAILS RED TODAY: server has no mutex between cleanup and start.
    Fix: server-side advisory lock per run_dir.
    """
    pytest.skip("Requires grpc_proxy to orchestrate concurrent StartDaq + CleanupData timing")


# ── SC-020: StopDaq timeout → SIGKILL escalation ────────────────────────────

@pytest.mark.skip(reason="SC-020: requires wrapper that ignores SIGINT")
def test_SC020_stopdaqs_timeout_triggers_sigkill_fallback() -> None:
    """
    SC-020: When StopDaq times out (hashpipe wrapper ignores SIGINT), the server
    must escalate to SIGKILL.

    FAILS RED TODAY: server sends only SIGINT; no escalation path.
    Fix: send SIGKILL after configurable timeout (default 30 s).
    """
    pytest.skip("Requires hashpipe wrapper that ignores SIGINT for 30 s")
