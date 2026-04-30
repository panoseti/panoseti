import os
import shutil
import time

import grpc
import pytest

from control.pseti import app
from control.utils import util

# ── Tests ───────────────────────────────────────────────────────────────────

def test_headnode_and_daq_space(daq_config, min_disk_gb):
    """
    Verify that the data directories specified in daq_config exist 
    and have sufficient free space on the physical SSD.
    """
    # Head Node
    head_dir = daq_config.head_node_data_dir
    print(f"Checking head node space: {head_dir}")
    assert os.path.exists(head_dir), f"Head node data dir {head_dir} does not exist!"
    
    usage = shutil.disk_usage(head_dir)
    free_gb = usage.free / (2**30)
    assert free_gb >= min_disk_gb, f"Head node disk space low: {free_gb:.1f}GB < {min_disk_gb}GB"

    # DAQ Nodes
    for node in daq_config.daq_nodes:
        daq_dir = node.data_dir
        ip_addr = str(node.ip_addr)
        print(f"Checking DAQ node ({ip_addr}) space: {daq_dir}")
        
        # In this HITL setup, we verify paths visible to the test runner machine.
        # This typically means these are network mounts or the test is running on the headnode.
        assert os.path.exists(daq_dir), f"DAQ node data dir {daq_dir} does not exist!"
        usage = shutil.disk_usage(daq_dir)
        free_gb = usage.free / (2**30)
        assert free_gb >= min_disk_gb, f"DAQ node {ip_addr} disk space low: {free_gb:.1f}GB"

def test_validate_commands(runner):
    """Ensure pseti validate passes schema and global checks."""
    result = runner.invoke(app, ["val"])
    assert result.exit_code == 0
    assert "✅ ALL VALIDATION CHECKS PASSED" in result.stdout

def test_network_ping_sweep(runner, daq_config, obs_config):
    """
    Verify physical network topology. 
    DAQ nodes must be up; Quabos must be down (initial state).
    """
    result = runner.invoke(app, ["val", "network"])
    assert result.exit_code == 0
    
    # Assert DAQ nodes are reachable
    for node in daq_config.daq_nodes:
        ip = str(node.ip_addr)
        assert ip in result.stdout
        assert f"DAQ Node ({ip})" in result.stdout

    # Assert Quabos are down (assuming they were powered off by safety net or initial state)
    # We get all valid Quabo IPs from the obs_config
    for dome in obs_config.domes:
        for module in dome.modules:
            base_ip = str(module.ip_addr)
            for i in range(4):
                # Resolve the quabo IP (might be base or PF-based)
                # But 'pseti validate network' reports them by their physical IP if not GW.
                # Actually, validate_network reports "Module (Dome: IP)"
                ip_parts = base_ip.split('.')
                quabo_ip = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.{int(ip_parts[3]) + i}"
                assert quabo_ip in result.stdout
                # We expect them to be down initially
                assert f"{quabo_ip} is DOWN" in result.stdout

def test_grpc_liveness(daq_config, network_config):
    """Verify pseti-grpc server is responding on Head and DAQ nodes."""
    # 1. Head Node Check
    head_ip = str(daq_config.head_node_ip_addr)
    head_port = 50051 # Default for headnode
    
    print(f"Checking Head Node gRPC: {head_ip}:{head_port}")
    channel = grpc.insecure_channel(f"{head_ip}:{head_port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=15)
    except grpc.FutureTimeoutError:
        pytest.fail(f"Head Node gRPC server on {head_ip}:{head_port} is not responding!")
    finally:
        channel.close()

    # 2. DAQ Node Checks
    # Attach network config to resolve endpoints with port forwarding if necessary
    util.attach_daq_config(daq_config, network_config)
    
    for node in daq_config.daq_nodes:
        host, port = util.daq_grpc_endpoint(node, daq_config)
        print(f"Checking DAQ Node gRPC: {host}:{port} (Physical: {node.ip_addr})")
        
        channel = grpc.insecure_channel(f"{host}:{port}")
        try:
            grpc.channel_ready_future(channel).result(timeout=15)
        except grpc.FutureTimeoutError:
            pytest.fail(f"DAQ Node gRPC server on {host}:{port} is not responding!")
        finally:
            channel.close()

def test_quabo_power_cycle(runner, obs_config, boot_wait_time):
    """
    Verify physical power control and boot sequence.
    1. Power On -> 2. Wait -> 3. Verify Ping -> 4. Power Off
    """
    print("Powering ON Quabos...")
    res_on = runner.invoke(app, ["power", "on"])
    assert res_on.exit_code == 0
    
    print(f"Waiting {boot_wait_time}s for Quabo boot...")
    time.sleep(boot_wait_time)
    
    print("Verifying Quabos are UP...")
    res_ping = runner.invoke(app, ["val", "network", "--yes"])
    assert res_ping.exit_code == 0
    
    for dome in obs_config.domes:
        for module in dome.modules:
            base_ip = str(module.ip_addr)
            for i in range(4):
                ip_parts = base_ip.split('.')
                quabo_ip = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.{int(ip_parts[3]) + i}"
                assert f"{quabo_ip} is UP" in res_ping.stdout

    print("Powering OFF Quabos...")
    #res_off = runner.invoke(app, ["power", "off", "--yes"])
    #assert res_off.exit_code == 0