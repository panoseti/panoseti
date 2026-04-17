#! /usr/bin/env python3

# collect files from remote DAQ nodes at the end of a recording run
#
# options when run as a cmdline script:
#
# --run_dir X   specify run dir
# --cleanup     clean up DAQ nodes; don't collect
# --verbose
import os
import sys
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import config_file, file_xfer, util
from utils.pydantic_config_models import DaqConfigValidator


# return '' if data collection was successful, else error msg
#
def collect_data(daq_config: DaqConfigValidator | dict[str, Any], run_dir: str, verbose: bool = False) -> str:
    if isinstance(daq_config, dict):
        daq_config = DaqConfigValidator(**daq_config)
    
    my_ip = util.local_ip()
    error_msg = ''
    for node in daq_config.daq_nodes:
        if not node.module_ids:
            continue
        # We need to know which module IDs are on this node
        for module_id in node.module_ids:
            if str(node.ip_addr) in my_ip:
                # head node is also a DAQ node.
                # Move files locally; if different volume, this will copy
                cmd = f"mv {node.data_dir}/module_{module_id}/{run_dir}/* {daq_config.head_node_data_dir}/{run_dir}"
                if verbose:
                    print(cmd)
                ret = os.system(cmd)
                if ret:
                    error_msg += f'command {cmd} failed: {ret}'
            else:
                error_msg += file_xfer.copy_dir_from_node(
                    run_dir, daq_config.model_dump(), node.model_dump(), int(module_id), verbose
                )
    return error_msg

# remove stuff from DAQ nodes no longer needed after run
# remote:
#    data/run
#    data/module_n/run
# local
#    data/module_n/run (should be empty dir)
# return error message or ''
#
def cleanup_daq(daq_config: DaqConfigValidator | dict[str, Any], run_dir: str, verbose: bool = False) -> str:
    if isinstance(daq_config, dict):
        daq_config = DaqConfigValidator(**daq_config)

    my_ip = util.local_ip()
    error_msg = ''
    for node in daq_config.daq_nodes:
        ip_addr = str(node.ip_addr)
        if ip_addr in my_ip:
            cmd = f'rm -rf {node.data_dir}/module_*/{run_dir}'
            if verbose:
                print(cmd)
            ret = os.system(cmd)
            if ret:
                error_msg += f'cleanup_daq(): {cmd} returned {ret} '
        else:
            rcmd = f'rm -rf {node.data_dir}/module_*/{run_dir}; rm -rf {node.data_dir}/{run_dir}'
            node_dict = node.model_dump()
            if 'port_forwarding' in node_dict:
                cmd = f"ssh -p {node_dict['port_forwarding']['port']} {node.username}@{node_dict['port_forwarding']['gw_ip']} \"{rcmd}\""
            else:
                cmd = f'ssh {node.username}@{ip_addr} "{rcmd}"'
            if verbose:
                print(cmd)
            ret = os.system(cmd)
            if ret:
                error_msg += f'cleanup_daq(): {cmd} returned {ret} '
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
