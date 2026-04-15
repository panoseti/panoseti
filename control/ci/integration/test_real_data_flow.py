"""
test_real_data_flow.py — Integration tests for the real hashpipe data path.

End-to-end pipeline under test:
    tcpreplay → UDP packets → hashpipe net_thread → UDS sockets →
    daq_data gRPC server → headnode streaming client

Requires:
    - Docker SDK (docker>=7.0) mounted at /var/run/docker.sock
    - tcpreplay installed inside the daqnode container
    - hashpipe.so at /data/hashpipe.so inside the daqnode container
    - PCAP file at /app/ci/integration/data/*.pcapng inside daqnode

With the unified server, daq_data and daq_control run in the same container
process, so hashpipe UDS sockets at /tmp are directly accessible to the
daq_data service — no shared volume needed.

The current PCAP contains pulse-height data only.  Tests request both
stream_movie_data and stream_pulse_height_data so that when mixed PH+MM
PCAP files are added in the future the tests remain valid without changes.

Function-scoped fixtures: each test gets a fresh hashpipe run.
"""
from __future__ import annotations

from itertools import islice

import pytest

from .conftest import (
    copy_run_dir,
    wait_hashpipe_stopped,
    wait_until,
)

# ---------------------------------------------------------------------------
# Guard: skip unless explicitly enabled
# ---------------------------------------------------------------------------

pytest.importorskip(
    "panoseti_grpc.daq_data.client",
    reason="panoseti_grpc.daq_data not available",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRealDataFlow:
    """End-to-end tests: tcpreplay → hashpipe → daq_data gRPC → headnode."""

    def test_hashpipe_writes_data_dirs(self, hashpipe_pcap_session, daq_data_dir):
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

    def test_real_stream_delivers_frames(self, hashpipe_pcap_session, real_daq_data_client):
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

    def test_frame_is_dict(self, hashpipe_pcap_session, real_daq_data_client):
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

    
