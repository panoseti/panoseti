"""
Hardware-facing mutations for `pseti start`: configuring Quabo data flow and
laying out run directories (head node + DAQ nodes).

Split out of start.py. `verbose` is an explicit parameter here rather than a
module-level global shared with start.py's CLI -- a cross-module mutable
global would silently stop working the moment these functions moved to a
different module (`from x import verbose` captures the value at import
time, not a live reference).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from ipaddress import ip_address
from pathlib import Path
from typing import TYPE_CHECKING

from panoseti_grpc.telemetry.logger import get_logger

from control.driver import quabo_driver
from control.utils import config_file, file_xfer, util
from control.utils.paths import PanoPaths

if TYPE_CHECKING:
    from control.utils.pydantic_config_models import DaqConfig, DataConfig, NetworkConfig, ObsConfig, QuaboUids

log_dir = PanoPaths.logs_dir()
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger(
    "PSETI.Start",
    log_dir=log_dir,
    grpc_enabled=True,
    reset=True
)


def start_data_flow(
    quabo_uids: QuaboUids,
    data_config: DataConfig,
    daq_config: DaqConfig,
    network_config: NetworkConfig,
    verbose: bool = False,
) -> None:
    """Initialize data flow from Quabos by configuring networking and modes.

    For every Quabo in every module:
    1. Tell it where to send Housekeeping (HK) packets (head node).
    2. Tell it where to send Data packets (assigned DAQ node).
    3. Set its DAQ acquisition mode (Image/PH/Stim/Flash).
    4. Synchronize PPS.

    Args:
        quabo_uids: Validated Quabo UID configuration.
        data_config: Science/engineering acquisition parameters.
        daq_config: DAQ node and head node networking details.
        network_config: Network routing and port forwarding settings.
        verbose: Print each command as it's issued.
    """
    daq_params = quabo_driver.get_daq_params(data_config)
    head_node_ip_addr = str(daq_config.head_node_ip_addr)

    def _configure_module(module: object) -> None:
        # Note: QuaboUidModule has 'ip_addr'
        base_ip_addr = str(module.ip_addr)  # type: ignore[attr-defined]
        module_id = config_file.ip_addr_to_module_id(base_ip_addr)
        try:
            daq_node = config_file.module_id_to_daq_node(daq_config, module_id)
        except Exception:
            return
        daq_node_ip_addr = str(daq_node.ip_addr)
        for i in range(4):
            if module.quabos[i].uid == '':  # type: ignore[attr-defined]
                continue
            ip_addr = config_file.quabo_ip_addr(base_ip_addr, i)
            ip_ports = util.get_quabo_ip_port(module.ip_addr, i, network_config)  # type: ignore[attr-defined]
            real_ip = ip_ports.ip_addr
            cmd_port = ip_ports.cmd_port
            logger.info(f'Quabo IP: {ip_addr}')
            logger.info(f'Real IP: {real_ip}')
            logger.info(f'Cmd Port: {cmd_port}')
            quabo = quabo_driver.QUABO(real_ip, cmd_port)
            if verbose:
                print(f'setting HK packet dest to {head_node_ip_addr} on quabo {ip_addr}')
            quabo.hk_packet_destination(ip_address(head_node_ip_addr))
            if verbose:
                print(f'setting data packet dest to {daq_node_ip_addr} on quabo {ip_addr}')
            quabo.data_packet_destination(ip_address(daq_node_ip_addr))
            if verbose:
                print(f'setting DAQ mode on quabo {ip_addr}')
            quabo.send_daq_params(daq_params)
            quabo.close()
        # send software 1PPS
        time.sleep(0.5)
        logger.info(f'Send software 1PPS to {base_ip_addr}')
        ip_ports = util.get_quabo_ip_port(module.ip_addr, 0, network_config)  # type: ignore[attr-defined]
        real_ip = ip_ports.ip_addr
        cmd_port = ip_ports.cmd_port
        quabo = quabo_driver.QUABO(real_ip, cmd_port)
        quabo.swpps()
        quabo.close()

    # Modules are independent (each has its own quabos, DAQ node assignment,
    # and WR/GNSS timing already established by session-start's reboot
    # sequence) -- configuring them one at a time was also actively working
    # against the point of the per-module software-PPS sync step above: at
    # more than one module, sequential sends meant the last module's
    # acquisition-start sync landed measurably later than the first's,
    # exactly the kind of cross-module skew a PPS sync exists to avoid.
    # Parallel, same pattern as config.py's do_reboot().
    all_modules = [module for dome in quabo_uids.domes for module in dome.modules]
    with ThreadPoolExecutor(max_workers=max(1, len(all_modules))) as pool:
        list(pool.map(_configure_module, all_modules))


def make_run_dirs(
    run_name: str,
    obs_config: ObsConfig,
    daq_config: DaqConfig,
    quabo_uids: QuaboUids,
    data_config: DataConfig,
    network_config: NetworkConfig,
    verbose: bool = False,
) -> None:
    """Create hierarchical run directories and snapshot configuration files.

    Snapshotting Contract:
    - Instead of copying from disk (which can mutate), this method writes the
      in-memory Pydantic models back to JSON files in the run directory.
    - Ensures the run directory is a faithful record of the actual run parameters.
    """
    run_dir = f'{daq_config.head_node_data_dir}/{run_name}'
    os.makedirs(run_dir, exist_ok=True)

    # 1. Snapshot in-memory config models to head node run dir
    # This prevents mid-flight disk mutations from leaking into the run records.
    # Note: we exclude bidirectional links ('modules', 'daq_node') to avoid circular serialization.
    config_snapshots = {
        config_file.obs_config_filename: obs_config,
        config_file.daq_config_filename: daq_config,
        config_file.data_config_filename: data_config,
        config_file.network_config_filename: network_config,
        config_file.quabo_uids_filename: quabo_uids,
    }

    for filename, model in config_snapshots.items():
        base_name = os.path.basename(filename)
        dest_path = f"{run_dir}/{base_name}"
        with open(dest_path, "w") as f:
            # We use model_dump then json.dump to allow fine-grained exclusion
            data = model.model_dump(exclude={'modules', 'daq_node'})
            json.dump(data, f, indent=4, default=str)

    # 2. Snapshot additional transient artifacts into the head node run dir.
    # We copy EVERYTHING from state/calibration to ensure full context is archived.
    calib_dir = PanoPaths.calibration_dir()
    if calib_dir.exists():
        for item in calib_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, Path(run_dir) / item.name)

    # Explicitly ensure sw_info.json and ph_baseline.json are captured from their primary locations
    artifact_map = {
        config_file.quabo_ph_baseline_filename: calib_dir / config_file.quabo_ph_baseline_filename,
        config_file.sw_info_filename: PanoPaths.tmp_dir() / config_file.sw_info_filename,
    }
    for base_name, src_path in artifact_map.items():
        if src_path and src_path.exists():
             shutil.copy2(src_path, f'{run_dir}/{base_name}')
        else:
             logger.debug(f"Artifact {base_name} not found in expected locations; skipping snapshot.")

    # 3. make module and run directories on DAQ nodes
    for node in daq_config.daq_nodes:
        # Check if this node has any modules assigned
        # DaqNode has module_ids
        if not node.module_ids:
            continue
        if util.is_local(node.ip_addr, daq_config):
            # We need to know which module IDs are on this node to create module_N dirs
            # node.module_ids is a list of ints or a range string (preprocessed to list[int])
            os.makedirs(f'{node.data_dir}/{run_name}', exist_ok=True)
            for mid in node.module_ids:
                path = f'{node.data_dir}/module_{mid}/{run_name}'
                if verbose:
                    print(f"mkdir -p {path}")
                os.makedirs(path, exist_ok=True)
        else:
            ip_addr = str(node.ip_addr)
            username = node.username
            data_dir = node.data_dir
            rcmds = [f'mkdir {data_dir}/{run_name}']
            for mid in node.module_ids:
                rcmds.append(f'mkdir -p {data_dir}/module_{mid}/{run_name}')
            # create process snapshot
            rcmds.append(f'cd {data_dir}/{run_name}; ps -ux > pss_{ip_addr}.log')
            rcmnd = ';'.join(rcmds)
            logger.info(f'DAQ IP: {ip_addr}')
            ssh_args = ["ssh", *util.ssh_options]
            if node.port_forwarding and node.port_forwarding.status:
                real_ip = str(node.port_forwarding.gw_ip)
                port = str(node.port_forwarding.port)
                ssh_args.extend(["-p", port, f"{username}@{real_ip}"])
            else:
                ssh_args.append(f"{username}@{ip_addr}")
            ssh_args.append(rcmnd)

            if verbose:
                print(" ".join(ssh_args))
            res = subprocess.run(ssh_args, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Failed to create run dirs on {ip_addr}: {res.stderr}")

    # copy config files to DAQ nodes
    file_xfer.copy_config_files(daq_config, run_name, verbose)
