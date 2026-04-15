"""
test_real_data_advanced.py — Rigorous integration tests for the real hashpipe data path.

Validates throughput, timing, consistency, and concurrency under real load.
"""
from __future__ import annotations

import os
import time
import asyncio

import pytest

from panoseti_grpc.daq_data.client import AioDaqDataClient
from .conftest import (
    REAL_HP_IO_CFG,
    DAQNODE_DATA_HOST,
    copy_run_dir,
    wait_hashpipe_stopped,
)

# pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
if not os.getenv("RUN_REAL_DATA_TESTS"):
    pytest.skip("Skipping real hashpipe data tests. Set RUN_REAL_DATA_TESTS=1", allow_module_level=True)


class TestRealDataAdvanced:

    @pytest.mark.asyncio
    async def test_real_data_starts_and_streams(self, hashpipe_pcap_session):
        """Basic validation: gRPC streams dictionaries successfully from real hashpipe UDS."""
        daq_config = {"daq_nodes": [{"ip_addr": DAQNODE_DATA_HOST}]}
        
        hp_io_cfg = {
            "data_dir": hashpipe_pcap_session["data_dir"],
            "update_interval_seconds": 0.1,
            "simulate_daq": False, 
            "force": True,
            "module_ids": [],
        }

        async with AioDaqDataClient(daq_config, network_config=None) as client:
            assert await client.init_hp_io(hosts=None, hp_io_cfg=hp_io_cfg), "Failed to init real DAQ"

            stream = await client.stream_images(
                hosts=None,
                stream_movie_data=True,
                stream_pulse_height_data=True,
                update_interval_seconds=0.1
            )

            received_frames = []
            try:
                async with asyncio.timeout(10.0):
                    async for frame in stream:
                        assert isinstance(frame, dict), f"Expected dict, got {type(frame)}"
                        assert len(frame) > 0, "Frame dict is empty"
                        received_frames.append(frame)
                        if len(received_frames) >= 3:
                            break
            except asyncio.TimeoutError:
                pytest.fail("Stream timed out before receiving 3 frames.")

            assert len(received_frames) == 3

    @pytest.mark.asyncio
    async def test_frame_arrival_timing_and_consistency(self, hashpipe_pcap_session):
        """Rigorous validation: Checks inter-frame timing and module_id consistency."""
        daq_config = {"daq_nodes": [{"ip_addr": DAQNODE_DATA_HOST}]}
        
        async with AioDaqDataClient(daq_config, network_config=None) as client:
            await client.init_hp_io(hosts=None, hp_io_cfg={
                "data_dir": hashpipe_pcap_session["data_dir"],
                "update_interval_seconds": 0.1,
                "simulate_daq": False,
                "force": True,
                "module_ids": []
            })

            stream = await client.stream_images(
                hosts=None, stream_movie_data=True, stream_pulse_height_data=True, update_interval_seconds=0.1
            )

            arrival_times = []
            frames = []
            
            try:
                async with asyncio.timeout(15.0):
                    async for frame in stream:
                        arrival_times.append(time.monotonic())
                        frames.append(frame)
                        if len(frames) >= 10:
                            break
            except asyncio.TimeoutError:
                pass

            assert len(frames) >= 10, f"Only {len(frames)} frames arrived within 15s. Is tcpreplay running?"

            # 1. Timing Validation: Ensure tcpreplay injection rate matches our stream expectations
            intervals = [arrival_times[i+1] - arrival_times[i] for i in range(len(arrival_times) - 1)]
            mean_interval = sum(intervals) / len(intervals)
            assert mean_interval < 2.0, f"Mean inter-frame interval {mean_interval:.2f}s is too large."

            # 2. Consistency Validation: Ensure module discovery is working
            module_ids = {f.get("module_id") for f in frames if "module_id" in f}
            assert len(module_ids) > 0, "Frames are missing module_ids"
            assert len(module_ids) <= 2, f"Expected 1-2 modules from test pcap, got {len(module_ids)}: {module_ids}"

    @pytest.mark.asyncio
    async def test_concurrent_streaming_stress(self, hashpipe_pcap_session):
        """Stress test: ensures the server handles multiple active streams of the same real data simultaneously."""
        daq_config = {"daq_nodes": [{"ip_addr": DAQNODE_DATA_HOST}]}

        async def receive_data(client_instance, required_frames=15):
            stream = await client_instance.stream_images(
                hosts=None, stream_movie_data=True, stream_pulse_height_data=True, update_interval_seconds=0.1
            )
            count = 0
            async with asyncio.timeout(15.0):
                async for _ in stream:
                    count += 1
                    if count >= required_frames:
                        break
            return count

        async with AioDaqDataClient(daq_config, network_config=None) as client1, \
                   AioDaqDataClient(daq_config, network_config=None) as client2:
            
            # Initialize DAQ on client1 (state is shared on the server)
            await client1.init_hp_io(hosts=None, hp_io_cfg={
                "data_dir": hashpipe_pcap_session["data_dir"],
                "update_interval_seconds": 0.1,
                "simulate_daq": False, "force": True, "module_ids": []
            })

            # Stream from both concurrently
            results = await asyncio.gather(
                receive_data(client1),
                receive_data(client2)
            )

            assert results[0] >= 15, f"Client 1 only received {results[0]} frames"
            assert results[1] >= 15, f"Client 2 only received {results[1]} frames"

    def test_data_collectible_after_stop(self, hashpipe_pcap_session, daq_control_direct, head_data_dir):
        """After StopDaq, data is copy-able to the headnode and cleanup succeeds."""
        run_params = hashpipe_pcap_session

        try:
            daq_control_direct.StopDaq({
                "data_dir": run_params["data_dir"],
                "run_dir":  run_params["run_dir"],
            })
            wait_hashpipe_stopped(daq_control_direct, run_params["data_dir"], timeout=8)
        except Exception:
            pass

        # Copy from shared volume (simulates rsync pulling to headnode)
        copy_ok = copy_run_dir(run_params, head_data_dir)
        assert copy_ok, "Data copy to headnode failed (no module dirs on daqnode?)"

        # Cleanup daqnode data after successful copy
        ok = daq_control_direct.CleanupData({
            "data_dir":  run_params["data_dir"],
            "run_dir":   run_params["run_dir"],
            "module_id": run_params["module_id"],
        })
        assert ok['success'], "CleanupData failed"