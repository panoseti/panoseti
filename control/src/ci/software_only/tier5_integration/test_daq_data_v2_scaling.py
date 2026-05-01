"""
test_daq_data_v2_scaling.py — Tier 5 scaling test for DaqData v2.
Verifies that multiple clients can receive streams of real Hashpipe data
from the centralized aggregator.
"""

import asyncio
import socket

import grpc
import pytest
from panoseti_grpc.daq_data_v2.client import AioDaqDataV2Client
from panoseti_grpc.daq_data_v2.server import DaqDataV2Servicer
from panoseti_grpc.generated import daq_data_v2_pb2_grpc

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(120)]

@pytest.fixture(scope="module")
async def aggregator_service():
    """Starts a local DaqDataV2 aggregator on the tester container."""
    import logging
    logger = logging.getLogger("aggregator")
    logger.setLevel(logging.INFO)
    servicer = DaqDataV2Servicer(logger)
    
    server = grpc.aio.server()
    daq_data_v2_pb2_grpc.add_DaqDataV2Servicer_to_server(servicer, server)
    
    # Use port 0 for dynamic allocation to avoid parallel worker collisions
    port = server.add_insecure_port("0.0.0.0:0")
    await server.start()
    
    # Identify the reachable IP for the forwarder (on the DAQ network)
    import psutil
    tester_ip = "127.0.0.1"
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address.startswith("172.25.0."):
                tester_ip = addr.address
                break
    
    if tester_ip == "127.0.0.1":
        # Fallback to hostname if logic fails
        tester_ip = socket.gethostbyname(socket.gethostname())
        
    target = f"{tester_ip}:{port}"
    print(f"DEBUG: Local aggregator started at {target}")
    
    async with AioDaqDataV2Client(target) as client:
        for _ in range(30):
            if await client.ping():
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail(f"Local aggregator at {target} failed to start")
            
    yield target, servicer
    await server.stop(0)

class TestDaqDataV2Scaling:
    """Verifies multi-client streaming of real science data."""

    async def test_v2_scaling_multi_client(self, daq_control_direct, aggregator_service, run_params, daqnode_container) -> None:
        """Verifies N clients receiving frames simultaneously."""
        NUM_CLIENTS = 10
        FRAMES_PER_CLIENT = 5
        agg_target, servicer = aggregator_service
        
        # 1. Start DAQ with v2 forwarder
        daqnode_container.exec_run("ip link set lo promisc on")
        
        params = dict(run_params)
        params["enable_v2_forwarder"] = True
        params["headnode_target"] = agg_target
        params["bindhost"] = "lo"
        
        # Clean start
        daq_control_direct.StopDaq({"data_dir": params["data_dir"], "run_dir": params["run_dir"]})
        await asyncio.sleep(1.0)
        
        print(f"DEBUG: Starting DAQ with forwarder pointing to {agg_target}")
        assert daq_control_direct.StartDaq(params) is True
        
        # 2. Start tcpreplay
        PCAP_DIR = "/app/src/ci/fixtures/data/"
        PCAP_GLOB = "*.pcapng"
        replay_cmd = f"sh -c 'tcpreplay --mbps=1.0 --loop=0 --intf1=lo {PCAP_GLOB}'"
        daqnode_container.exec_run(replay_cmd, detach=True, workdir=PCAP_DIR)
        
        # Diagnostic: Wait for frames to reach aggregator cache
        frames_arrived = False
        for i in range(50):
            if servicer.cache:
                frames_arrived = True
                print(f"DEBUG: Frames reached aggregator after {i}s")
                break
            await asyncio.sleep(1.0)
        
        if not frames_arrived:
            _, logs = daqnode_container.exec_run("cat /var/log/panoseti/daq_control_server.log")
            print(f"--- DAQ NODE LOGS ---\n{logs.decode()}")
            pytest.fail("Frames never arrived at aggregator cache.")
        
        async def client_task(cid: int):
            async with AioDaqDataV2Client(agg_target) as client:
                received = 0
                async for response in client.stream_images(update_interval=0.01):
                    received += 1
                    if received >= FRAMES_PER_CLIENT:
                        break
                return cid, received

        # 3. Spawn concurrent clients
        tasks = [asyncio.create_task(client_task(i)) for i in range(NUM_CLIENTS)]
        
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=60.0)
            for cid, count in results:
                assert count >= FRAMES_PER_CLIENT, f"Client {cid} only received {count} frames"
        except TimeoutError:
            pytest.fail("Scaling test timed out waiting for clients to receive frames")
        finally:
            # Cleanup
            daqnode_container.exec_run("pkill -9 tcpreplay")
            daq_control_direct.StopDaq({"data_dir": params["data_dir"], "run_dir": params["run_dir"]})
