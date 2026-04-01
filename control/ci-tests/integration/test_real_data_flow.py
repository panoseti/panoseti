"""
test_real_data_flow.py — Integration tests for the real hashpipe data path.

End-to-end pipeline under test:
    tcpreplay → UDP packets → hashpipe net_thread → UDS sockets →
    daq_data gRPC server → headnode streaming client

Requires:
    - RUN_REAL_DATA_TESTS=1 env var (guard, skipped by default)
    - Docker SDK (docker>=7.0) mounted at /var/run/docker.sock
    - tcpreplay installed inside the daqnode container
    - hashpipe.so at /data/hashpipe.so inside the daqnode container
    - PCAP file at /app/ci-tests/integration/data/*.pcapng inside daqnode

With the unified server, daq_data and daq_control run in the same container
process, so hashpipe UDS sockets at /tmp are directly accessible to the
daq_data service — no shared volume needed.

The current PCAP contains pulse-height data only.  Tests request both
stream_movie_data and stream_pulse_height_data so that when mixed PH+MM
PCAP files are added in the future the tests remain valid without changes.

Function-scoped fixtures: each test gets a fresh hashpipe run.
"""
from __future__ import annotations

import os
import time
import pathlib
from itertools import islice

import pytest

from .conftest import (
    DAQ_DATA_DIR,
    DAQNODE_DATA_HOST,
    copy_run_dir,
    wait_hashpipe_running,
    wait_hashpipe_stopped,
    wait_until,
)

# ---------------------------------------------------------------------------
# Guard: skip unless explicitly enabled
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_DATA_TESTS") != "1",
    reason="Set RUN_REAL_DATA_TESTS=1 to enable PCAP-based hashpipe tests",
)

pytest.importorskip(
    "panoseti_grpc.daq_data.client",
    reason="panoseti_grpc.daq_data not available",
)

from panoseti_grpc.daq_data.client import DaqDataClient  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to PCAP file *inside* the daqnode container (after COPY . .)
_PCAP_GLOB = "/app/ci-tests/integration/data/*.pcapng"

# hp_io_cfg for real (non-simulated) hashpipe mode
_HP_IO_CFG = {
    "update_interval_seconds": 0.1,
    "simulate_daq": False,
    "force": True,
    "module_ids": [],   # stream from all active modules
}

_HASHPIPE_READY_RETRIES = 20


# ---------------------------------------------------------------------------
# Module-scoped fixture: start hashpipe + tcpreplay, tear down after all tests
# ---------------------------------------------------------------------------



@pytest.fixture
def hashpipe_pcap_session(daqnode_container, daq_control_direct, run_params):
    """
    Start hashpipe via daq_control gRPC, inject PCAP packets via docker exec
    tcpreplay, then yield.  Tears down hashpipe on exit.

    Function-scoped: each test gets its own fresh hashpipe run so tests are
    fully independent (test_data_collectible_after_stop stops hashpipe mid-test).
    """
    # 0. Verify PCAP exists so tcpreplay doesn't silently fail
    if daqnode_container.exec_run(f"sh -c 'ls {_PCAP_GLOB}'").exit_code != 0:
        pytest.fail(f"PCAP missing in container at {_PCAP_GLOB}")
    
    # 1. Start hashpipe via gRPC (bindhost=eth0 so it receives loopback packets)
    lp = {**run_params, "bindhost": "eth0"}
    try:
        daq_control_direct.StartDaq(lp)
    except Exception as e:
        pytest.fail(f"Failed to start hashpipe via gRPC: {e}")

    # 2. Wait for hashpipe to be confirmed running
    if not wait_hashpipe_running(
        daq_control_direct, run_params["data_dir"], timeout=_HASHPIPE_READY_RETRIES
    ):
        pytest.fail(f"hashpipe did not start within {_HASHPIPE_READY_RETRIES}s")
    
    # Forces the native Linux veth to accept the foreign MAC addresses from the PCAP
    daqnode_container.exec_run("ip link set eth0 promisc on")

    # 3. Run tcpreplay inside daqnode container (loop=5, low rate to avoid overflow)
    replay_cmd = f"sh -c 'tcpreplay --mbps=0.01 --loop=0 --intf1=eth0 {_PCAP_GLOB}'"
    # daqnode_container.exec_run(replay_cmd, detach=True)
    daqnode_container.exec_run(replay_cmd, detach=True)

    yield run_params
    
    # 4. Teardown
    # Kill TCPREPLAY first to stop the packet flood
    daqnode_container.exec_run("pkill -9 tcpreplay", detach=False)

    # 5. Teardown
    try:
        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
    except Exception:
        pass
    assert wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"], timeout=8)


# ---------------------------------------------------------------------------
# Helper: daq_data client configured for real (non-simulated) mode
# ---------------------------------------------------------------------------

@pytest.fixture
def real_daq_data_client(hashpipe_pcap_session, ensure_clean_daq_state):
    """
    DaqDataClient connected to the unified daqnode gRPC server.
    daq_data and daq_control share a process, so hashpipe UDS sockets
    at /tmp are directly accessible — no shared volume required.
    """
    run_params = hashpipe_pcap_session
    daq_cfg = {
        "daq_nodes": [{"ip_addr": DAQNODE_DATA_HOST, "data_dir": run_params["data_dir"]}]
    }
    with DaqDataClient(daq_cfg, network_config=None) as client:
        ok = client.init_hp_io(hosts=None, hp_io_cfg=_HP_IO_CFG)
        if not ok:
            pytest.skip(
                "init_hp_io(simulate_daq=False) failed — "
                "check that hashpipe started and UDS sockets are present at /tmp."
            )
        yield client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRealDataFlow:
    """End-to-end tests: tcpreplay → hashpipe → daq_data gRPC → headnode."""

    def test_hashpipe_writes_data_dirs(self, hashpipe_pcap_session, daq_data_dir, ensure_clean_daq_state):
        """
        After tcpreplay, hashpipe creates module-level data directories
        under DAQ_DATA_DIR/module_{id}/{run_dir}/.
        """
        run_params = hashpipe_pcap_session
        found = wait_until(
            lambda: any(
                (daq_data_dir / f"module_{mid}" / run_params["run_dir"]).exists()
                for mid in run_params["module_id"]
            ),
            timeout=30,
            interval=0.1,
        )
        if found:
            return
        pytest.fail(
            f"No module data directory appeared in {daq_data_dir} within 30s. "
            f"Expected module_{run_params['module_id']} / {run_params['run_dir']}"
        )

    def test_real_stream_delivers_frames(self, hashpipe_pcap_session, real_daq_data_client, ensure_clean_daq_state):
        """
        After init_hp_io(simulate_daq=False), stream_images() yields at least 1 frame
        driven by live hashpipe output (from tcpreplay PCAP injection).
        """
        run_params = hashpipe_pcap_session
        frames = list(islice(
            real_daq_data_client.stream_images(
                hosts=None,
                stream_movie_data=True,
                stream_pulse_height_data=True,
                update_interval_seconds=0.1,
            ),
            5,
        ))
        assert len(frames) >= 1, (
            "stream_images() yielded no frames from real hashpipe. "
            "Check tcpreplay ran and hashpipe UDS sockets are accessible."
        )

    def test_frame_is_dict(self, hashpipe_pcap_session, real_daq_data_client, ensure_clean_daq_state):
        """Each frame returned by the real stream is a non-empty dict."""
        run_params = hashpipe_pcap_session
        for frame in islice(
            real_daq_data_client.stream_images(
                hosts=None,
                stream_movie_data=True,
                stream_pulse_height_data=True,
                update_interval_seconds=0.1,
            ),
            3,
        ):
            assert isinstance(frame, dict), f"Expected dict, got {type(frame)}"
            assert len(frame) > 0, "Frame dict is empty"

    def test_data_collectible_after_stop(
        self,
        hashpipe_pcap_session,
        daq_control_direct,
        daq_data_dir,
        head_data_dir,
        ensure_clean_daq_state,
    ):
        """After StopDaq, data is copy-able to the headnode and cleanup succeeds."""
        run_params = hashpipe_pcap_session

        # Stop hashpipe (best-effort; fixture also does this on teardown)
        try:
            daq_control_direct.StopDaq({
                "data_dir": run_params["data_dir"],
                "run_dir":  run_params["run_dir"],
            })
            wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"], timeout=8)
        except Exception:
            pass

        # Copy from shared volume (simulates rsync)
        copy_ok = copy_run_dir(run_params, head_data_dir)
        assert copy_ok, "Data copy to headnode failed (no module dirs on daqnode?)"

        # Cleanup daqnode data after successful copy
        ok = daq_control_direct.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })['success']
        assert ok is True

    
