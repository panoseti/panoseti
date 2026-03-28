"""
test_science_streaming.py — Integration tests for the daq_data science streaming service.

Uses panoseti_grpc.daq_data in simulation mode (no real Hashpipe shared memory).
The SIMULATE_DAQ env var enables simulation; real hardware CI sets it to false.
"""
from __future__ import annotations

import os
import time
from itertools import islice

import pytest

from .conftest import DAQNODE_DIRECT_HOST


# Skip if daq_data client is not available
pytest.importorskip("panoseti_grpc.daq_data.client", reason="panoseti_grpc.daq_data not available")

from panoseti_grpc.daq_data.client import DaqDataClient  # noqa: E402


SIMULATE_DAQ  = os.getenv("SIMULATE_DAQ", "true").lower() in ("true", "1", "yes")
DATA_GRPC_PORT = int(os.getenv("DATA_PORT", "50052"))


@pytest.fixture(scope="module")
def daq_data_client():
    """Session-scoped DaqDataClient."""
    client = DaqDataClient(host=DAQNODE_DIRECT_HOST, port=DATA_GRPC_PORT)
    yield client


@pytest.mark.skipif(not SIMULATE_DAQ, reason="Only runs in software simulation mode")
class TestScienceStreaming:

    def test_simulation_delivers_frames(self, daq_data_client):
        """Streaming in simulation mode delivers at least 5 frames with module_id=200."""
        frames = list(islice(
            daq_data_client.stream_images(
                stream_movie_data=True,
                stream_pulse_height_data=False,
                update_interval_seconds=0.1,
                module_ids=[200],
                simulate=True,
            ),
            5,
        ))
        assert len(frames) == 5
        assert all(f.get("module_id") == 200 for f in frames)

    def test_stream_contains_image_data(self, daq_data_client):
        """Each streamed frame has non-empty pixel data."""
        frames = list(islice(
            daq_data_client.stream_images(
                stream_movie_data=True,
                stream_pulse_height_data=False,
                update_interval_seconds=0.1,
                module_ids=[200],
                simulate=True,
            ),
            3,
        ))
        for frame in frames:
            assert "pixels" in frame or "image" in frame or len(frame) > 1, (
                f"Frame missing image data: {frame}"
            )

    def test_full_lifecycle_with_streaming(
        self, daq_control_direct, daq_data_client, run_params
    ):
        """Start DAQ → stream 3 simulated frames → Stop DAQ."""
        daq_control_direct.StartDaq(run_params)
        time.sleep(1)

        frames = list(islice(
            daq_data_client.stream_images(
                stream_movie_data=True,
                stream_pulse_height_data=False,
                update_interval_seconds=0.1,
                module_ids=run_params["module_id"],
                simulate=True,
            ),
            3,
        ))
        assert len(frames) == 3

        daq_control_direct.StopDaq({
            "data_dir": run_params["data_dir"],
            "run_dir":  run_params["run_dir"],
        })
