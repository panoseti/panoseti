"""
test_real_data_extended.py — Advanced integration tests for the real hashpipe data path.

Validates gateway routing, client disconnect resilience, and state machine robustness
using the real tcpreplay -> hashpipe -> UDS pipeline.
"""
from __future__ import annotations

import asyncio

import pytest
from panoseti_grpc.daq_data.client import AioDaqDataClient

from ci.tier3_fleet.conftest import (
    DAQNODE_DATA_HOST,
    DAQNODE_GATEWAY_HOST,
)

pytestmark = pytest.mark.asyncio

# if not os.getenv("RUN_REAL_DATA_TESTS"):
#     pytest.skip("Skipping real hashpipe data tests. Set RUN_REAL_DATA_TESTS=1", allow_module_level=True)


class TestRealDataExtended:

    async def test_real_stream_via_gateway(self, hashpipe_pcap_session):
        """
        Validates that real data streams successfully through the socat gateway.
        This proves the data pipeline works for remote/NAT topologies, not just direct IP.
        """
        # Note: Targeting the GATEWAY IP instead of the direct IP
        daq_config = {"daq_nodes": [{"ip_addr": DAQNODE_GATEWAY_HOST}]}
        
        hp_io_cfg = {
            "data_dir": hashpipe_pcap_session["data_dir"],
            "update_interval_seconds": 0.1,
            "simulate_daq": False, 
            "force": True,
            "module_ids": [],
        }

        async with AioDaqDataClient(daq_config, network_config=None) as client:
            assert await client.init_hp_io(hosts=None, hp_io_cfg=hp_io_cfg), "Failed to init via gateway"

            stream = await client.stream_images(
                hosts=None,
                stream_movie_data=True,
                stream_pulse_height_data=True,
                update_interval_seconds=0.1
            )

            received = 0
            async with asyncio.timeout(15.0):
                async for frame in stream:
                    assert isinstance(frame, dict)
                    received += 1
                    if received >= 5:
                        break
            
            assert received >= 5, "Failed to stream 5 frames through the gateway"

    async def test_client_abrupt_disconnect_resilience(self, hashpipe_pcap_session):
        """
        Simulates a client crashing mid-stream. Ensures the server cleans up the
        active UDS socket readers and allows a new client to connect seamlessly.
        """
        daq_config = {"daq_nodes": [{"ip_addr": DAQNODE_DATA_HOST}]}
        hp_io_cfg = {
            "data_dir": hashpipe_pcap_session["data_dir"],
            "update_interval_seconds": 0.1,
            "simulate_daq": False, 
            "force": True,
            "module_ids": [],
        }

        # Client 1 connects, reads 2 frames, and abruptly drops context (simulating a crash)
        async with AioDaqDataClient(daq_config, network_config=None) as client1:
            await client1.init_hp_io(hosts=None, hp_io_cfg=hp_io_cfg)
            stream1 = await client1.stream_images(
                hosts=None, stream_movie_data=True, stream_pulse_height_data=True, update_interval_seconds=0.1
            )
            async with asyncio.timeout(5.0):
                await anext(stream1)
                await anext(stream1)
            # Exiting the 'async with' abruptly closes the gRPC channel

        # Give the server a fraction of a second to reap the broken gRPC connection
        await asyncio.sleep(0.5)

        # Client 2 connects. If the UDS socket is locked by a zombie reader, this will hang or fail.
        async with AioDaqDataClient(daq_config, network_config=None) as client2:
            stream2 = await client2.stream_images(
                hosts=None, stream_movie_data=True, stream_pulse_height_data=True, update_interval_seconds=0.1
            )
            received = 0
            async with asyncio.timeout(10.0):
                async for _frame in stream2:
                    received += 1
                    if received >= 5:
                        break
            
            assert received >= 5, "Client 2 failed to stream after Client 1 disconnected"

    async def test_double_init_real_daq_succeeds(self, hashpipe_pcap_session):
        """
        Validates idempotency. Calling init_hp_io while a real DAQ session is already
        initialized and streaming should safely tear down and rebuild the server-side
        pipeline without crashing.
        """
        daq_config = {"daq_nodes": [{"ip_addr": DAQNODE_DATA_HOST}]}
        hp_io_cfg = {
            "data_dir": hashpipe_pcap_session["data_dir"],
            "update_interval_seconds": 0.1,
            "simulate_daq": False, 
            "force": True,
            "module_ids": [],
        }

        async with AioDaqDataClient(daq_config, network_config=None) as client:
            # First init
            assert await client.init_hp_io(hosts=None, hp_io_cfg=hp_io_cfg)

            # Second init immediately after
            assert await client.init_hp_io(hosts=None, hp_io_cfg=hp_io_cfg)

            # Prove we can still get data
            stream = await client.stream_images(
                hosts=None, stream_movie_data=True, stream_pulse_height_data=True, update_interval_seconds=0.1
            )
            
            async with asyncio.timeout(5.0):
                frame = await anext(stream)
                assert frame is not None

    async def test_rapid_subscription_cycling(self, hashpipe_pcap_session):
        """
        Stresses the server's stream management. Rapidly creates and destroys stream
        subscriptions to ensure no memory leaks or thread deadlocks occur.
        """
        daq_config = {"daq_nodes": [{"ip_addr": DAQNODE_DATA_HOST}]}
        hp_io_cfg = {
            "data_dir": hashpipe_pcap_session["data_dir"],
            "update_interval_seconds": 0.1,
            "simulate_daq": False, 
            "force": True,
            "module_ids": [],
        }

        async with AioDaqDataClient(daq_config, network_config=None) as client:
            await client.init_hp_io(hosts=None, hp_io_cfg=hp_io_cfg)

            for cycle in range(5):
                stream = await client.stream_images(
                    hosts=None, stream_movie_data=True, stream_pulse_height_data=True, update_interval_seconds=0.1
                )
                
                # Get exactly 1 frame then immediately cancel the stream
                try:
                    async with asyncio.timeout(3.0):
                        await anext(stream)
                except TimeoutError:
                    pytest.fail(f"Stream cycle {cycle} timed out")
                
                # In Python's gRPC aio, cancelling the async generator closes the stream
                await stream.aclose()  # type: ignore[attr-defined]