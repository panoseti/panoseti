#! /usr/bin/env python3

# copy files to/from daq nodes
#
# options when run as a script:
# --config              copy data products config file to nodes
# --hashpipe            copy hashpipe executable (hashpipe.so) to nodes
# --get_data run_dir    copy data files in given run dir from daq nodes

import os
import subprocess
import sys
import time
from glob import glob

from control.utils import config_file, util
from control.utils.pydantic_config_models import DaqConfigValidator, DaqNodeValidator


# copy a file to a DAQ node
#
def copy_file_to_node(file: str, node: DaqNodeValidator, run_dir: str = '', verbose: bool = False) -> None:
    """Transfer local files to a remote DAQ node using SCP.

    Args:
        file: Glob pattern or path of files to transfer.
        node: The target DAQ node validator.
        run_dir: Optional subdirectory on the remote node.
        verbose: If True, prints the SCP command.

    Raises:
        Exception: If the SCP command fails.
    """
    dest_path = node.data_dir
    if run_dir:
        dest_path += f'/{run_dir}'
    else:
        dest_path += '/'
    files = glob(file)
    for f in files:
        if node.port_forwarding and node.port_forwarding.status:
            cmd = ["scp", "-q", "-P", str(node.port_forwarding.port), f, f"{node.username}@{node.port_forwarding.gw_ip}:{dest_path}"]
        else:
            cmd = ["scp", "-q", f, f"{node.username}@{node.ip_addr}:{dest_path}"]
        if verbose:
            print(" ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f'{" ".join(cmd)} returned {res.returncode}: {res.stderr}')

def _run_rsync_with_retry(cmd: list[str], verbose: bool = False) -> str:
    """Run rsync with exponential backoff retries for transient failures (Task 2.7)."""
    transient_codes = {12, 23, 30, 35, 255}
    max_retries = 3
    base_delay = 5
    for attempt in range(max_retries + 1):
        if verbose:
            print(" ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return ""
        
        if res.returncode in transient_codes and attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            print(f"Transient rsync error {res.returncode}. Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
            time.sleep(delay)
            continue
        
        err_type = "Transient" if res.returncode in transient_codes else "Terminal"
        return f"{err_type} rsync error {res.returncode}: {res.stderr}"
    return "Max retries exceeded"

# Copy the contents of a module/run dir from a DAQ node
# to the corresponding run dir on this (head) node.
# scp doesn't let you do this directly,
# so we copy the dir to a temp directory (data/IP_ADDR/run),
# then move (rename) the files into the target dir
#
# return error message, or '' on success
#
def copy_dir_from_node(run_name: str, daq_config: DaqConfigValidator, node: DaqNodeValidator, module_id: int, verbose: bool = False) -> str:
    """Synchronize observation data from a remote module directory to the head node.

    Uses rsync to pull Hashpipe output, process snapshots, and PFF data files.

    Args:
        run_name: Name of the current observation run.
        daq_config: Validated DAQ configuration.
        node: The target remote DAQ node validator.
        module_id: ID of the module whose data is being collected.
        verbose: If True, prints rsync commands.

    Returns:
        An empty string if successful, otherwise an error message.
    """
    local_data_dir = daq_config.head_node_data_dir
    run_dir_path = f'{local_data_dir}/{run_name}'

    if not os.path.isdir(run_dir_path):
        return f'copy_dir_from_node(): no run dir {run_dir_path}'

    pf = node.port_forwarding
    use_pf = pf is not None and pf.status

    # Base rsync command
    base_rsync = ["rsync", "-P"]
    if use_pf and pf is not None:
        base_rsync.extend(["-e", f"ssh -p {pf.port}"])
        remote_host = f"{node.username}@{pf.gw_ip}"
    else:
        remote_host = f"{node.username}@{node.ip_addr}"

    # Combined rsync call for stdout, snapshot, and PFF files (Phase 1 Optimization)
    cmd = [
        *base_rsync,
        f"{remote_host}:{node.data_dir}/{run_name}/{util.hp_stdout_prefix}*",
        f"{remote_host}:{node.data_dir}/{run_name}/{util.pss_prefix}*",
        f"{remote_host}:{node.data_dir}/module_{module_id}/{run_name}/*",
        run_dir_path
    ]
    err = _run_rsync_with_retry(cmd, verbose)
    if err:
        return f"copy_dir_from_node(combined): {err}"

    return ''

# create a directory on DAQ nodes
#
def make_remote_dirs(daq_config: DaqConfigValidator, dirname: str) -> None:
    """Create a directory on all configured remote DAQ nodes via SSH.

    Args:
        daq_config: Validated DAQ configuration.
        dirname: Path of the directory to create.

    Raises:
        Exception: If the SSH command fails on any node.
    """
    for node in daq_config.daq_nodes:
        cmd = ["ssh", f"{node.username}@{node.ip_addr}", f"cd {node.data_dir}; mkdir {dirname}"]
        print(" ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f'{" ".join(cmd)} returned {res.returncode}: {res.stderr}')

# copy config files to run dirs on DAQ nodes
#
def copy_config_files(daq_config: DaqConfigValidator, run_dir: str, verbose: bool = False) -> None:
    """Distribute all observatory configuration files to remote DAQ nodes.

    Args:
        daq_config: Validated DAQ configuration.
        run_dir: Name of the target run directory on the remote nodes.
        verbose: If True, prints transfer details.
    """
    for node in daq_config.daq_nodes:
        for f in config_file.config_file_names:
            copy_file_to_node(f, node, run_dir, verbose)

# copy hashpipe binary and scripts to data dirs on DAQ nodes
#
def copy_daq_files(daq_config: DaqConfigValidator) -> None:
    """Bootstrap remote DAQ nodes with essential software and scripts.

    Copies the Hashpipe binary (if present) and support scripts required
    for remote data acquisition.

    Args:
        daq_config: Validated DAQ configuration model.
    """
    # hashpipe.so may not exist, as we may cross compile it on the daq node
    hashpipe_so = '../daq/hashpipe.so'
    if os.path.exists(hashpipe_so):
        hashpipe_so_exist = True
    else:
        hashpipe_so_exist = False
        print('**************************************************************************')
        print('{} does not exist!'.format('hashpipe.so'))
        print('clone the submodule and compile it, or compile it on the daq node.')
        print('**************************************************************************')
    for node in daq_config.daq_nodes:
        if hashpipe_so_exist:
            copy_file_to_node(hashpipe_so, node)
        copy_file_to_node('daq_scripts/start_daq.py', node)
        copy_file_to_node('daq_scripts/stop_daq.py', node)
        copy_file_to_node('daq_scripts/status_daq.py', node)
        copy_file_to_node('utils/util.py', node)
        copy_file_to_node('utils/pff.py', node)
        copy_file_to_node('daq_scripts/video_daq.py', node)

if __name__ == "__main__":

    def usage() -> None:
        print('''options:
    --init_daq_nodes: copy software to DAQ nodes
    ''')
        sys.exit()

    argv = sys.argv
    do_init_daq_nodes = False
    i = 1
    while i < len(argv):
        if argv[i] == '--init_daq_nodes':
            do_init_daq_nodes = True
        else:
            usage()
        i += 1

    if not do_init_daq_nodes:
        usage()

    daq_config = config_file.get_daq_config()
    if do_init_daq_nodes:
        copy_daq_files(daq_config)
