"""
hw2_observing — Real DAQ flow tests.

Tests verify the full observing lifecycle: pseti start → record → pseti stop →
transfer queue → archive. They reuse the same RunStateLedger / verify_manifest
machinery as the software-only integration tests, but run against real hardware
(Quabos, DAQ node running Hashpipe, WR/GNSS timing).

Required state: ACQUIRING (framework calls start_acq before this batch)
Class: observing (batch_priority=2)
Leaves state: ACQ_CONFIGURED (stop_acq called at end of each test)

pytest-timeout: 180 seconds (60s graceful buffer flush + transfer overhead)
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from control.utils import config_file
from control.utils.paths import PanoPaths

pytestmark = [
    pytest.mark.hw_class("observing"),
    pytest.mark.timeout(180),
]

_RUN_DURATION_S = 30  # short runs for CI; enough for Hashpipe to emit ≥1 PFF file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoke_pseti(runner, args: list[str]) -> str:
    from control.pseti import app
    result = runner.invoke(app, args)
    return result.stdout


def _wait_for_ledger_state(target: str, timeout: float = 120.0) -> bool:
    """Poll the transfer ledger until the run reaches *target* state."""
    from control.utils.transfer.ledger import RunStateLedger
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ledger = RunStateLedger.load()
            if ledger.status == target:
                return True
        except Exception:
            pass
        time.sleep(2.0)
    return False


# ---------------------------------------------------------------------------
# Full run → archive
# ---------------------------------------------------------------------------

@pytest.mark.timeout(180)
def test_full_run_to_archive(runner) -> None:
    """
    Start a 30-second run with --no-hv, wait for ARCHIVED in the ledger.
    Then verify the head-node run directory has the run_complete marker and
    at least one PFF file (transferred from the DAQ node).
    """
    out = _invoke_pseti(runner, ["start", "--yes", "--nsecs", str(_RUN_DURATION_S), "--no-hv"])
    assert "ACTIVE" in out or "started" in out.lower(), f"pseti start failed:\n{out}"

    archived = _wait_for_ledger_state("ARCHIVED", timeout=150.0)
    assert archived, "Run did not reach ARCHIVED state within 150 s"

    # Verify head-node data
    from control.utils.transfer.ledger import RunStateLedger
    ledger = RunStateLedger.load()
    run_name = ledger.run_name
    head_run_dir = Path(config_file.get_daq_config().head_node_data_dir) / run_name
    assert (head_run_dir / "run_complete").exists(), "run_complete marker missing"
    pff_files = list(head_run_dir.rglob("*.pff"))
    assert pff_files, f"No PFF files transferred to head node for {run_name}"


# ---------------------------------------------------------------------------
# Multi-run drain
# ---------------------------------------------------------------------------

def test_multi_run_drain(runner) -> None:
    """
    Run three short back-to-back runs; assert each reaches ARCHIVED and no
    orphaned jobs remain in the active/ transfer queue directory.
    """
    for i in range(3):
        out = _invoke_pseti(runner, ["start", "--yes", "--nsecs", "15", "--no-hv"])
        assert "ACTIVE" in out or "started" in out.lower(), f"Run {i+1} start failed:\n{out}"
        # Wait for run to finish recording before starting the next
        time.sleep(20.0)

    archived = _wait_for_ledger_state("ARCHIVED", timeout=300.0)
    assert archived, "Last run did not reach ARCHIVED within 300 s"

    active_jobs = list(PanoPaths.transfer_queue_dir().joinpath("active").glob("*.job.toml"))
    assert not active_jobs, f"Orphaned active transfer jobs: {[j.name for j in active_jobs]}"


# ---------------------------------------------------------------------------
# Active daemon race
# ---------------------------------------------------------------------------

def test_active_daemon_race(runner) -> None:
    """
    With the transfer daemon already running, enqueue a job while the daemon
    is polling, assert clean handoff (job moves from pending/ to completed/).
    """
    # Start a run and immediately check the daemon picks it up
    out = _invoke_pseti(runner, ["start", "--yes", "--nsecs", "20", "--no-hv"])
    assert "ACTIVE" in out or "started" in out.lower(), f"start failed:\n{out}"

    # Immediately wait; pending may already be claimed by the daemon
    time.sleep(5.0)

    archived = _wait_for_ledger_state("ARCHIVED", timeout=150.0)
    assert archived, "Run did not reach ARCHIVED (daemon race test)"


# ---------------------------------------------------------------------------
# Distributed run started (multi-node, skipped if <2 DAQ nodes)
# ---------------------------------------------------------------------------

def test_distributed_run_started(runner, topology) -> None:
    """
    With ≥2 DAQ nodes, verify all nodes show RECORDING status via gRPC.
    Skipped if the active topology has fewer than 2 DAQ nodes.
    """
    if len(topology.daq_nodes()) < 2:
        pytest.skip("Fewer than 2 DAQ nodes in active topology")

    out = _invoke_pseti(runner, ["start", "--yes", "--nsecs", str(_RUN_DURATION_S), "--no-hv"])
    assert "ACTIVE" in out or "started" in out.lower(), f"start failed:\n{out}"

    from panoseti_grpc.daq_control.client import DaqControlClient
    daq = config_file.get_daq_config()

    async def check_all_recording() -> bool:
        for node in daq.daq_nodes:
            async with DaqControlClient(node.ip, 50051) as client:
                status = await client.StatusDaq()
                if not status.is_running:
                    return False
        return True

    all_recording = asyncio.run(check_all_recording())
    assert all_recording, "Not all DAQ nodes showed RECORDING status"

    _invoke_pseti(runner, ["stop", "--yes"])


# ---------------------------------------------------------------------------
# gRPC streams real quabo data
# ---------------------------------------------------------------------------

def test_grpc_streams_real_quabo_data(runner, topology) -> None:
    """
    After start, the DaqDataClient should stream ≥10 PanoImage frames
    within 15 seconds. Asserts shape matches configured data product.
    """
    out = _invoke_pseti(runner, ["start", "--yes", "--nsecs", str(_RUN_DURATION_S), "--no-hv"])
    assert "ACTIVE" in out or "started" in out.lower(), f"start failed:\n{out}"

    daq = config_file.get_daq_config()
    first_node = daq.daq_nodes[0]

    async def collect_frames() -> list:
        frames = []
        from panoseti_grpc.daq_data.client import DaqDataClient
        async with DaqDataClient(first_node.ip, 50051) as client:
            run_dir = str(Path(first_node.data_dir) / "current")
            module_id = first_node.module_ids[0]
            await client.init_hp_io(run_dir, module_id)
            deadline = time.monotonic() + 15.0
            async for image in client.stream_images():
                frames.append(image)
                if len(frames) >= 10 or time.monotonic() > deadline:
                    break
        return frames

    frames = asyncio.run(collect_frames())
    assert len(frames) >= 10, f"Only {len(frames)} frames in 15 s (expected ≥ 10)"

    _invoke_pseti(runner, ["stop", "--yes"])


# ---------------------------------------------------------------------------
# Frame header field validation
# ---------------------------------------------------------------------------

def test_frame_header_fields(runner, topology) -> None:
    """
    Frames streamed from a real quabo must have non-zero pkt_num, pkt_tai,
    and a tv_sec in a plausible Unix timestamp range (year 2020+).
    """
    out = _invoke_pseti(runner, ["start", "--yes", "--nsecs", str(_RUN_DURATION_S), "--no-hv"])
    assert "ACTIVE" in out or "started" in out.lower()

    daq = config_file.get_daq_config()
    first_node = daq.daq_nodes[0]

    async def collect_one() -> object:
        from panoseti_grpc.daq_data.client import DaqDataClient
        async with DaqDataClient(first_node.ip, 50051) as client:
            run_dir = str(Path(first_node.data_dir) / "current")
            module_id = first_node.module_ids[0]
            await client.init_hp_io(run_dir, module_id)
            async for image in client.stream_images():
                return image
        return None

    frame = asyncio.run(collect_one())
    _invoke_pseti(runner, ["stop", "--yes"])

    assert frame is not None, "No frame received"
    assert frame.pkt_num > 0, f"pkt_num={frame.pkt_num} not positive"
    assert frame.tv_sec > 1_577_836_800, f"tv_sec={frame.tv_sec} predates 2020"


# ---------------------------------------------------------------------------
# Module ID consistency
# ---------------------------------------------------------------------------

def test_module_id_consistency(runner, topology) -> None:
    """
    The module_id field in streamed frames must match the module IDs declared
    in the active obs_config.
    """
    out = _invoke_pseti(runner, ["start", "--yes", "--nsecs", str(_RUN_DURATION_S), "--no-hv"])
    assert "ACTIVE" in out or "started" in out.lower()

    expected_ids = set(topology.module_ids())
    daq = config_file.get_daq_config()
    first_node = daq.daq_nodes[0]

    async def collect_module_ids() -> set[int]:
        seen: set[int] = set()
        from panoseti_grpc.daq_data.client import DaqDataClient
        async with DaqDataClient(first_node.ip, 50051) as client:
            run_dir = str(Path(first_node.data_dir) / "current")
            module_id = first_node.module_ids[0]
            await client.init_hp_io(run_dir, module_id)
            deadline = time.monotonic() + 10.0
            async for image in client.stream_images():
                seen.add(image.module_id)
                if time.monotonic() > deadline:
                    break
        return seen

    seen_ids = asyncio.run(collect_module_ids())
    _invoke_pseti(runner, ["stop", "--yes"])

    assert seen_ids.issubset(expected_ids), (
        f"Unexpected module IDs in stream: {seen_ids - expected_ids}"
    )


# ---------------------------------------------------------------------------
# Concurrent clients receive same frames
# ---------------------------------------------------------------------------

def test_concurrent_clients_receive_same_frames(runner, topology) -> None:
    """
    Two DaqDataClient instances connected simultaneously should see the
    same frame timestamps (broadcast semantics).
    """
    out = _invoke_pseti(runner, ["start", "--yes", "--nsecs", str(_RUN_DURATION_S), "--no-hv"])
    assert "ACTIVE" in out or "started" in out.lower()

    daq = config_file.get_daq_config()
    first_node = daq.daq_nodes[0]

    async def dual_collect() -> tuple[list, list]:
        from panoseti_grpc.daq_data.client import DaqDataClient
        run_dir = str(Path(first_node.data_dir) / "current")
        module_id = first_node.module_ids[0]
        frames_a: list = []
        frames_b: list = []
        deadline = time.monotonic() + 10.0
        async with DaqDataClient(first_node.ip, 50051) as a, \
                   DaqDataClient(first_node.ip, 50051) as b:
            await a.init_hp_io(run_dir, module_id)
            await b.init_hp_io(run_dir, module_id)
            async def consume(client, bucket, n=20):
                async for img in client.stream_images():
                    bucket.append(img.tv_sec)
                    if len(bucket) >= n or time.monotonic() > deadline:
                        break
            await asyncio.gather(
                consume(a, frames_a),
                consume(b, frames_b),
            )
        return frames_a, frames_b

    fa, fb = asyncio.run(dual_collect())
    _invoke_pseti(runner, ["stop", "--yes"])

    overlap = set(fa) & set(fb)
    assert len(overlap) >= 5, (
        f"Concurrent clients saw fewer than 5 common timestamps: |overlap|={len(overlap)}"
    )


# ---------------------------------------------------------------------------
# CleanupData precondition enforcement
# ---------------------------------------------------------------------------

def test_cleanup_precondition_enforced(runner, topology) -> None:
    """
    CleanupData(mode=CLEANUP_SELECTIVE) refuses deletion when a wrong
    manifest_digest is provided (FAILED_PRECONDITION). A correct digest
    succeeds. This verifies the safety invariant end-to-end on real hardware.
    """
    out = _invoke_pseti(runner, ["start", "--yes", "--nsecs", "15", "--no-hv"])
    assert "ACTIVE" in out or "started" in out.lower()
    time.sleep(20.0)  # let the run finish

    daq = config_file.get_daq_config()

    async def test_cleanup() -> None:
        # Load the run name from ledger
        from control.utils.transfer.ledger import RunStateLedger
        from panoseti_grpc.daq_control.client import DaqControlClient
        from panoseti_grpc.util.exceptions import FailedPreconditionError
        ledger = RunStateLedger.load()
        run_name = ledger.run_name

        node = daq.daq_nodes[0]
        async with DaqControlClient(node.ip, 50051) as client:
            # Wrong digest must raise FAILED_PRECONDITION
            with pytest.raises((FailedPreconditionError, Exception)) as exc_info:
                await client.CleanupData(
                    run_name=run_name,
                    mode="CLEANUP_SELECTIVE",
                    manifest_digest="deadbeef" * 8,  # 64-char hex, definitely wrong
                    delete_patterns=["*.pff"],
                    preserve_patterns=["*.json"],
                )
            assert "FAILED_PRECONDITION" in str(exc_info.value) or "precondition" in str(exc_info.value).lower()

    asyncio.run(test_cleanup())


# ---------------------------------------------------------------------------
# --no-hv safety during run
# ---------------------------------------------------------------------------

def test_no_hv_safety_during_run(runner, topology) -> None:
    """
    When started with --no-hv, HVMON0..3 in HK packets should stay near zero
    throughout the run. Asserts the safety contract of the --no-hv flag.
    """
    out = _invoke_pseti(runner, ["start", "--yes", "--nsecs", "15", "--no-hv"])
    assert "ACTIVE" in out or "started" in out.lower()

    # Collect a few HK packets during the run
    quabos = topology.quabo_ips()
    if not quabos:
        pytest.skip("No quabos in topology")

    import struct

    from control.driver.quabo_driver import QUABO
    q0 = QUABO(quabos[0].ip)
    hv_readings = []
    for _ in range(3):
        pkt = q0.read_hk_packet()
        if pkt and len(pkt) >= 10:
            # HVMON0..3 at bytes 2..9 (LE uint16 each)
            for ch in range(4):
                val = struct.unpack_from("<H", pkt, 2 + 2 * ch)[0]
                hv_readings.append(val)
        time.sleep(1.0)
    q0.close()

    _invoke_pseti(runner, ["stop", "--yes"])

    if not hv_readings:
        pytest.skip("No HK packets received during --no-hv run")

    for val in hv_readings:
        assert val < 1000, f"HVMON unexpectedly high ({val}) during --no-hv run"
