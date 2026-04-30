#!/usr/bin/env python3
"""
ci/tier5_setup.py — Maintainable preparation script for Tier 5 Integration tests.
Handles config copying, IP shifting, and environment validation.
"""

import os
import pathlib
import shutil
import sys


def setup_integration_configs():
    # 1. Paths
    root = pathlib.Path(__file__).parent.parent.resolve()
    src_dir = root / "ci" / "fixtures" / "configs" / "direct"
    dst_dir = pathlib.Path("/tmp/pseti_test/integration_configs")
    tmp_dir = pathlib.Path("/tmp/pseti_test/tmp")
    logs_dir = pathlib.Path("/tmp/pseti_test/logs")
    
    # 2. Subnet Prefixes (from environment)
    head_prefix = os.getenv("HEAD_NET_PREFIX", "10.51.1")
    daq_prefix = os.getenv("DAQ_NET_PREFIX", "172.25.0")
    quabo_prefix = os.getenv("QUABO_NET_PREFIX", "172.25.3")
    
    print(f"Setting up Tier 5 configs with prefixes: HEAD={head_prefix}, DAQ={daq_prefix}, QUABO={quabo_prefix}")

    # 3. Create directories
    for d in [dst_dir, tmp_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
        # Recursive chmod 777 for container access
        os.chmod(d, 0o777)

    # 4. Copy and Shift IPs
    if not src_dir.exists():
        print(f"Error: Source config directory {src_dir} not found.")
        sys.exit(1)

    for src_file in src_dir.glob("*.json"):
        dst_file = dst_dir / src_file.name
        content = src_file.read_text()
        
        # Shift IPs while preserving Module IDs (octet 3 and 4)
        content = content.replace("192.168.0.", f"{daq_prefix}.")
        content = content.replace("192.168.3.", f"{quabo_prefix}.")
        content = content.replace("10.0.1.", f"{head_prefix}.")
        
        dst_file.write_text(content)
        os.chmod(dst_file, 0o666)
        print(f"  Processed: {src_file.name} -> {dst_file}")

    # 6. Coherence Fix: Ensure daq_config handles all modules in obs_config
    import json
    obs_path = dst_dir / "obs_config.json"
    daq_path = dst_dir / "daq_config.json"
    if obs_path.exists() and daq_path.exists():
        obs = json.loads(obs_path.read_text())
        daq = json.loads(daq_path.read_text())
        
        mids = []
        # Shift module IPs in obs_config to match prefixes
        for dome in obs.get("domes", []):
            for module in dome.get("modules", []):
                old_ip = module.get("ip_addr")
                if old_ip:
                    # Maintain octets 3 and 4, shift 1 and 2
                    parts = old_ip.split(".")
                    new_ip = f"{quabo_prefix}.{parts[3]}"
                    module["ip_addr"] = new_ip
                    # Recalculate ID
                    n = int(parts[3]) + 256*int(parts[2])
                    mids.append((n>>2)&255)
        
        obs_path.write_text(json.dumps(obs, indent=4))
        print(f"  Coherence: Updated obs_config modules to {quabo_prefix}.x")

        if daq.get("daq_nodes"):
            # Assign all found modules to the first node
            daq["daq_nodes"][0]["module_ids"] = list(set(mids))
            daq["daq_nodes"][0]["ip_addr"] = f"{daq_prefix}.10"
            daq_path.write_text(json.dumps(daq, indent=4))
            print(f"  Coherence: Assigned modules {mids} to {daq['daq_nodes'][0]['ip_addr']}")

    # 5. Provide quabo_uids.json
    uids_src = root / "ci" / "fixtures" / "configs" / "quabo_uids_chaos.json"
    uids_dst = tmp_dir / "quabo_uids.json"
    if uids_src.exists():
        if uids_dst.exists() or uids_dst.is_symlink():
            uids_dst.unlink()
        shutil.copy(uids_src, uids_dst)
        os.chmod(uids_dst, 0o666)
        print(f"  Copied: {uids_src.name} -> {uids_dst}")

if __name__ == "__main__":
    setup_integration_configs()
