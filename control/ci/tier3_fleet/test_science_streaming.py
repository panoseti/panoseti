"""
test_science_streaming.py — Integration tests for the daq_data science streaming service.

Simulation path only: tests call init_sim() which makes the daq_data server stream from
a bundled PFF file. This isolates the gRPC → headnode path without real hashpipe hardware.

Each test calls init_sim() independently — do NOT share server state between tests.
For real tcpreplay → hashpipe → gRPC → headnode tests, see test_real_data_flow.py.
"""
from __future__ import annotations

from itertools import islice

import pytest

from ci.tier3_fleet.conftest import DAQNODE_DATA_HOST

pytest.importorskip(
    "panoseti_grpc.daq_data.client",
    reason="panoseti_grpc.daq_data not available",
)


class TestScienceStreamingSimulation:
    """Tests the daq_data gRPC server in simulation mode (init_sim → stream_images)."""

    def test_ping(self, daq_data_client, ensure_clean_daq_state) -> None:
        """daq_data server at daqnode-data is reachable."""
        hosts = daq_data_client.get_valid_daq_hosts()
        assert len(hosts) >= 1, (
            f"No valid daq_data hosts found. Expected {DAQNODE_DATA_HOST} to be reachable."
        )

    def test_init_sim(self, daq_data_client, ensure_clean_daq_state) -> None:
        """init_sim() succeeds — server configures itself to stream from bundled PFF."""
        ok = daq_data_client.init_sim(hosts=None)
        assert ok is True

    def test_stream_delivers_frames(self, daq_data_client, ensure_clean_daq_state) -> None:
        """After init_sim(), stream_images() yields at least 1 frame."""
        assert daq_data_client.init_sim(hosts=None) is True
        frames = list(islice(
            daq_data_client.stream_images(
                hosts=None,
                stream_movie_data=True,
                stream_pulse_height_data=True,
                update_interval_seconds=0.1,
            ),
            3,
        ))
        assert len(frames) >= 1, "stream_images() yielded no frames after init_sim()"

    def test_frame_is_dict(self, daq_data_client, ensure_clean_daq_state) -> None:
        """Each streamed frame is a non-empty dict (parsed PanoImage)."""
        assert daq_data_client.init_sim(hosts=None) is True
        for frame in islice(
            daq_data_client.stream_images(
                hosts=None,
                stream_movie_data=True,
                stream_pulse_height_data=True,
                update_interval_seconds=0.1,
            ),
            3,
        ):
            assert isinstance(frame, dict), f"Expected dict frame, got {type(frame)}"
            assert len(frame) > 0, "Frame dict is empty"

    def test_double_init_sim_succeeds(self, daq_data_client, ensure_clean_daq_state) -> None:
        """Calling init_sim() twice (force reconfigure) does not raise."""
        assert daq_data_client.init_sim(hosts=None) is True
        assert daq_data_client.init_sim(hosts=None) is True
