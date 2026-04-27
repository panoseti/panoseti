"""
test_integration_real_data.py — Tier 5 Heavy Integration tests for real Hashpipe flows.

Pipeline: tcpreplay → UDP → hashpipe → UDS → daq_data gRPC → client.
"""
from __future__ import annotations

import os
import pathlib
from itertools import islice

import pytest

from ci.tier3_fleet.conftest import (
    copy_run_dir,
    wait_hashpipe_stopped,
    wait_until,
)


@pytest.mark.usefixtures("hashpipe_pcap_session")
class TestIntegrationRealDataFlow:
    """End-to-end tests: tcpreplay → hashpipe → daq_data gRPC → headnode."""

    def test_hashpipe_writes_data_dirs(self, run_params) -> None:
        """Hashpipe creates module-level data directories under DAQ_DATA_DIR."""
        host_data_root = os.environ.get("DAQ_DATA_DIR", "/data")
        daq_data_path = pathlib.Path(host_data_root)
        
        found = wait_until(
            lambda: any(
                (daq_data_path / f"module_{mid}" / run_params["run_dir"]).exists()
                for mid in run_params["module_id"]
            ),
            timeout=30,
            interval=0.1,
        )
        assert found, f"No module data directory appeared in {host_data_root}"

    def test_real_stream_delivers_frames(self, daq_data_client, hashpipe_pcap_session) -> None:
        """stream_images() yields at least 1 frame driven by live hashpipe output."""
        import time

        import grpc
        
        # Capped retry loop for Hashpipe initialization (SC-055 resolution)
        MAX_RETRIES = 30
        RETRY_INTERVAL = 1.0
        frames = []
        
        for i in range(MAX_RETRIES):
            try:
                frames = list(islice(
                    daq_data_client.stream_images(
                        hosts=None,
                        stream_movie_data=True,
                        stream_pulse_height_data=True,
                        update_interval_seconds=0.1,
                    ),
                    5,
                ))
                if frames:
                    break
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.FAILED_PRECONDITION and "hp_io" in e.details():
                    # Expected transient error while hashpipe initializes threads
                    time.sleep(RETRY_INTERVAL)
                    continue
                raise
            except Exception:
                raise

        assert len(frames) >= 1, f"Failed to deliver frames after {MAX_RETRIES} retries."

    def test_frame_is_dict(self, daq_data_client, hashpipe_pcap_session) -> None:
        """Each frame returned by the real stream is a non-empty dict."""
        import time

        import grpc
        
        MAX_RETRIES = 30
        RETRY_INTERVAL = 1.0
        frames = []
        
        for i in range(MAX_RETRIES):
            try:
                frames = list(islice(
                    daq_data_client.stream_images(
                        hosts=None,
                        stream_movie_data=True,
                        stream_pulse_height_data=True,
                        update_interval_seconds=0.1,
                    ),
                    3,
                ))
                if frames:
                    break
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.FAILED_PRECONDITION and "hp_io" in e.details():
                    time.sleep(RETRY_INTERVAL)
                    continue
                raise
            except Exception:
                raise

        assert len(frames) >= 1, f"Failed to deliver frames after {MAX_RETRIES} retries."
        for frame in frames:
            assert isinstance(frame, dict)
            assert len(frame) > 0

    def test_data_collectible_after_stop(
        self,
        daq_control_direct,
        run_params,
        head_data_dir,
    ) -> None:
        """After StopDaq, data is copy-able to the headnode and cleanup succeeds."""
        # Stop hashpipe
        daq_control_direct.StopDaq({
            "data_dir": "/data",
            "run_dir":  run_params["run_dir"],
        })
        wait_hashpipe_stopped(daq_control_direct, "/data", timeout=8)

        # Copy from shared volume
        assert copy_run_dir(run_params, pathlib.Path(head_data_dir))

        # Cleanup daqnode data
        result = daq_control_direct.CleanupData({
            "data_dir":  "/data",
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
            "force": True
        })
        assert result.get("success") is True
