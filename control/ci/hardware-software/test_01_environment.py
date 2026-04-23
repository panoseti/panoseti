import os
import time
import shutil
import pytest
import socket
import grpc
from pathlib import Path
from control.pseti import app

# ── Tests ───────────────────────────────────────────────────────────────────

def test_headnode_and_daq_space(daq_config, min_disk_gb):
    """
    Verify that the data directories specified in daq_config.json 
    exist and have sufficient free space on the physical SSD.
    """
    # Head Node
    head_dir = daq_config["head_node_data_dir"]
    print(f"Checking head node space: {head_dir}")
    assert os.path.exists(head_dir), f"Head node data dir {head_dir} does not exist!"
    
    usage = shutil.disk_usage(head_dir)
    free_gb = usage.free / (2**30)
    assert free_gb >= min_disk_gb, f"Head node disk space low: {free_gb:.1f}GB < {min_disk_gb}GB"

    # DAQ Node
    # Note: In HITL, the DAQ node is remote but mounts the same /mnt/panoseti path locally or via NFS.
    # We assume the test runner has visibility or we verify the path mapping.
    for node in daq_config["daq_nodes"]:
        daq_dir = node["data_dir"]
        print(f"Checking DAQ node ({node['ip_addr']}) space: {daq_dir}")
        # If the DAQ node is the same as headnode (IP check), we already checked it.
        # Otherwise, for physical HITL, we assume the path is locally mounted for verification
        # or we rely on 'pseti config disk-space' in a real scenario.
        # For this test, we verify the path exists on the machine running the tests.
        assert os.path.exists(daq_dir), f"DAQ node data dir {daq_dir} does not exist!"
        usage = shutil.disk_usage(daq_dir)
        free_gb = usage.free / (2**30)
        assert free_gb >= min_disk_gb, f"DAQ node {node['ip_addr']} disk space low: {free_gb:.1f}GB"

def test_validate_commands(runner):
    """Ensure pseti validate passes schema and global checks."""
    result = runner.invoke(app, ["validate", "--yes"])
    assert result.exit_code == 0
    assert "Validation successful" in result.stdout

def test_network_ping_sweep(runner):
    """
    Verify physical network topology. 
    DAQ node must be up; Quabos must be down (initial state).
    """
    result = runner.invoke(app, ["validate", "network", "--yes"])
    assert result.exit_code == 0
    
    # Assert DAQ node is reachable
    # Expected output contains status of nodes
    assert "192.168.0.228" in result.stdout
    assert "UP" in result.stdout.upper()
    
    # Assert Quabos are down (assuming they were powered off by safety net or initial state)
    # 192.168.3.248-251
    for i in range(248, 252):
        ip = f"192.168.3.{i}"
        # The network validate output should show them as DOWN
        # pseti validate network output usually lists IPs and their status
        assert ip in result.stdout
        # We expect them to be down initially
        # Finding the line for this IP and checking status
        for line in result.stdout.splitlines():
            if ip in line:
                assert "DOWN" in line.upper()

def test_grpc_liveness():
    """Verify panoseti-server is responding on Head and DAQ nodes."""
    nodes = ["192.168.88.103", "192.168.0.228"]
    port = 50051
    
    for ip in nodes:
        print(f"Checking gRPC liveness for {ip}:{port}")
        # Basic socket check for the port first
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            assert s.connect_ex((ip, port)) == 0, f"gRPC port {port} not open on {ip}"
        
        # Actual gRPC connection attempt (generic channel check)
        channel = grpc.insecure_channel(f"{ip}:{port}")
        try:
            # We use a 5s timeout to ensure the server is actually responding
            grpc.channel_ready_future(channel).result(timeout=5)
        except grpc.FutureTimeoutError:
            pytest.fail(f"gRPC server on {ip}:{port} is not responding!")
        finally:
            channel.close()

def test_quabo_power_cycle(runner, boot_wait_time):
    """
    Verify physical power control and boot sequence.
    1. Power On -> 2. Wait -> 3. Verify Ping -> 4. Power Off
    """
    print("Powering ON Quabos...")
    res_on = runner.invoke(app, ["power", "on", "--yes"])
    assert res_on.exit_code == 0
    
    print(f"Waiting {boot_wait_time}s for Quabo boot...")
    time.sleep(boot_wait_time)
    
    print("Verifying Quabos are UP...")
    res_ping = runner.invoke(app, ["validate", "network", "--yes"])
    assert res_ping.exit_code == 0
    
    for i in range(248, 252):
        ip = f"192.168.3.{i}"
        assert ip in res_ping.stdout
        for line in res_ping.stdout.splitlines():
            if ip in line:
                assert "UP" in line.upper()

    print("Powering OFF Quabos...")
    res_off = runner.invoke(app, ["power", "off", "--yes"])
    assert res_off.exit_code == 0
