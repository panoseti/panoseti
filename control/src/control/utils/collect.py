#! /usr/bin/env python3

# collect files from remote DAQ nodes at the end of a recording run
#
# options when run as a cmdline script:
#
# --run_dir X   specify run dir
# --cleanup     clean up DAQ nodes; don't collect
# --verbose
import shutil
import subprocess
import sys
import warnings
from glob import glob

warnings.warn(
    "utils.collect is deprecated. The transfer daemon (utils.transfer.daemon) "
    "now owns all data collection. Use TransferQueue.enqueue() to schedule transfers.",
    DeprecationWarning,
    stacklevel=2,
)


from control.utils import config_file, file_xfer, util  # noqa: E402
from control.utils.pydantic_config_models import CollectResult, DaqConfig  # noqa: E402


# return CollectResult if data collection was successful
#
def collect_data(daq_config: DaqConfig, run_dir: str, verbose: bool = False) -> CollectResult:
    """Aggregate PFF data files from remote DAQ nodes to the local head node.

    Uses rsync/SCP or local move (if head node is a DAQ node) to centralize
    artifacts into the hierarchical run directory structure.

    Args:
        daq_config: Validated DAQ configuration model.
        run_dir: Name of the current observation run directory.
        verbose: If True, prints detailed file transfer commands.

    Returns:
        A CollectResult object containing success status and error messages.
    """
    errors = []
    failed_ips = set()
    for node in daq_config.daq_nodes:
        if not node.module_ids:
            continue
        # We need to know which module IDs are on this node
        for module_id in node.module_ids:
            if util.is_local(node.ip_addr, daq_config):
                # head node is also a DAQ node — move files locally.
                src_pattern = f"{node.data_dir}/module_{module_id}/{run_dir}/*"
                dst = f"{daq_config.head_node_data_dir}/{run_dir}"
                if verbose:
                    print(f"mv {src_pattern} {dst}")
                for src_file in glob(src_pattern):
                    try:
                        shutil.move(src_file, dst)
                    except Exception as exc:
                        msg = f"Local move failed for module {module_id}: {exc}"
                        errors.append(msg)
                        failed_ips.add(str(node.ip_addr))
            else:
                err = file_xfer.copy_dir_from_node(
                    run_dir, daq_config, node, int(module_id), verbose
                )
                if err:
                    errors.append(err)
                    failed_ips.add(str(node.ip_addr))
    
    return CollectResult(
        success=len(errors) == 0,
        errors=errors,
        failed_ips=list(failed_ips)
    )

# remove stuff from DAQ nodes no longer needed after run
# remote:
#    data/run
#    data/module_n/run
# local
#    data/module_n/run (should be empty dir)
# return error message or ''
#
def cleanup_daq(daq_config: DaqConfig, run_dir: str, verbose: bool = False) -> str:
    """Remove observation artifacts from DAQ nodes after successful collection.

    Deletes the run-specific directories in both the root data path and
    per-module subdirectories on each remote node.

    Args:
        daq_config: Validated DAQ configuration model.
        run_dir: Name of the run directory to clean up.
        verbose: If True, prints removal commands.

    Returns:
        An empty string if successful, otherwise a combined error message.
    """
    error_msg = ''
    for node in daq_config.daq_nodes:
        ip_addr = str(node.ip_addr)
        if util.is_local(node.ip_addr, daq_config):
            path = f'{node.data_dir}/module_*/{run_dir}'
            cmd = f'rm -rf {path}'
            if verbose:
                print(cmd)
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode != 0:
                error_msg += f'cleanup_daq() local failed: {res.stderr} '
        else:
            rcmd = f'rm -rf {node.data_dir}/module_*/{run_dir}; rm -rf {node.data_dir}/{run_dir}'
            ssh_args = ["ssh"]
            if node.port_forwarding and node.port_forwarding.status:
                ssh_args.extend(["-p", str(node.port_forwarding.port), f"{node.username}@{node.port_forwarding.gw_ip}"])
            else:
                ssh_args.append(f"{node.username}@{ip_addr}")
            ssh_args.append(rcmd)
            
            if verbose:
                print(" ".join(ssh_args))
            res = subprocess.run(ssh_args, capture_output=True, text=True)
            if res.returncode != 0:
                error_msg += f'cleanup_daq() remote {ip_addr} failed: {res.stderr} '
    return error_msg

if __name__ == "__main__":
    i = 1
    run_dir = ''
    verbose = False
    cleanup = False
    while i<len(sys.argv):
        if sys.argv[i] == '--run_dir':
            i += 1
            run_dir = sys.argv[i]
        elif sys.argv[i] == '--verbose':
            verbose = True
        elif sys.argv[i] == '--cleanup':
            cleanup = True
        i += 1
    if not run_dir:
        run_dir_val = util.read_run_name()
        if not run_dir_val:
            raise Exception("No run found")
        run_dir = run_dir_val
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    config_file.associate(daq_config, quabo_uids)
    if cleanup:
        cleanup_daq(daq_config, run_dir, verbose)
    else:
        ret = collect_data(daq_config, run_dir, verbose)
        print('success' if not ret else 'failed')
