#! /usr/bin/env python3

# copy files to/from daq nodes
#
# options when run as a script:
# --config              copy data products config file to nodes
# --hashpipe            copy hashpipe executable (hashpipe.so) to nodes
# --get_data run_dir    copy data files in given run dir from daq nodes

import os
import sys
from glob import glob
from typing import Any

from utils import config_file, util
from utils.pydantic_config_models import DaqConfigValidator


# copy a file to a DAQ node
#
def copy_file_to_node(file: str, daq_config: dict[str, Any], node: dict[str, Any], run_dir: str = '', verbose: bool = False) -> None:
    """Transfer local files to a remote DAQ node using SCP.

    Args:
        file: Glob pattern or path of files to transfer.
        daq_config: Raw DAQ configuration dictionary.
        node: Raw dictionary for the target DAQ node.
        run_dir: Optional subdirectory on the remote node.
        verbose: If True, prints the SCP command.

    Raises:
        Exception: If the SCP command fails.
    """
    # TODO: daq_config is not used
    dest_path = node['data_dir']
    if run_dir:
        dest_path += f'/{run_dir}'
    else:
        dest_path += '/'
    files = glob(file)
    for f in files:
        if 'port_forwarding' in node:
            cmd = f"scp -q -P {node['port_forwarding']['port']} {f} {node['username']}@{node['port_forwarding']['gw_ip']}:{dest_path}"
        else:
            cmd = 'scp -q {} {}@{}:{}'.format(
                f, node['username'], node['ip_addr'], dest_path
            )
        if verbose:
            print(cmd)
        ret = os.system(cmd)
        if ret:
            raise Exception(f'{cmd} returned {ret}')

# Copy the contents of a module/run dir from a DAQ node
# to the corresponding run dir on this (head) node.
# scp doesn't let you do this directly,
# so we copy the dir to a temp directory (data/IP_ADDR/run),
# then move (rename) the files into the target dir
#
# return error message, or '' on success
#
def copy_dir_from_node(run_name: str, daq_config: dict[str, Any], node: dict[str, Any], module_id: int, verbose: bool = False) -> str:
    """Synchronize observation data from a remote module directory to the head node.
    
    Uses rsync to pull Hashpipe output, process snapshots, and PFF data files.

    Args:
        run_name: Name of the current observation run.
        daq_config: Raw DAQ configuration dictionary.
        node: Raw dictionary for the target remote node.
        module_id: ID of the module whose data is being collected.
        verbose: If True, prints rsync commands.

    Returns:
        An empty string if successful, otherwise an error message.
    """
    local_data_dir = daq_config['head_node_data_dir']
    run_dir_path = f'{local_data_dir}/{run_name}'

    if not os.path.isdir(run_dir_path):
        return f'copy_dir_from_node(): no run dir {run_dir_path}'
        
    # copy stdout from remote node to this node
    if 'port_forwarding' in node:
        cmd = f'rsync -P -e "ssh -p {node["port_forwarding"]["port"]}" {node["username"]}@{node["port_forwarding"]["gw_ip"]}:{node["data_dir"]}/{run_name}/{util.hp_stdout_prefix}* {run_dir_path}'
    else:
        cmd = 'rsync -P {}@{}:{}/{}/{}* {}'.format(
            node['username'], node['ip_addr'],
            node['data_dir'], run_name, util.hp_stdout_prefix,
            run_dir_path
        )
    if verbose:
        print(cmd)
    try:
        ret = os.system(cmd)
        if ret:
            return f'copy_dir_from_node(): {cmd} returned {ret}'
    except Exception as e:
        return f'copy_dir_from_node(): {cmd} failed with {e}'
    
    # copy process snapshot from remote node to this node
    if 'port_forwarding' in node:
        cmd = f'rsync -P -e "ssh -p {node["port_forwarding"]["port"]}" {node["username"]}@{node["port_forwarding"]["gw_ip"]}:{node["data_dir"]}/{run_name}/{util.pss_prefix}* {run_dir_path}'
    else:
        cmd = 'rsync -P {}@{}:{}/{}/{}* {}'.format(
            node['username'], node['ip_addr'],
            node['data_dir'], run_name, util.pss_prefix,
            run_dir_path
        )
    if verbose:
        print(cmd)
    try:
        ret = os.system(cmd)
        if ret:
            return f'copy_dir_from_node(): {cmd} returned {ret}'
    except Exception as e:
        return f'copy_dir_from_node(): {cmd} failed with {e}'
    # copy PFF files from remote node to this node
    if 'port_forwarding' in node:
        cmd = f'rsync -P -e "ssh -p {node["port_forwarding"]["port"]}" {node["username"]}@{node["port_forwarding"]["gw_ip"]}:{node["data_dir"]}/module_{module_id}/{run_name}/* {run_dir_path}'
    else:
        cmd = f'rsync -P {node["username"]}@{node["ip_addr"]}:{node["data_dir"]}/module_{module_id}/{run_name}/* {run_dir_path}'
    if verbose:
        print(cmd)
    try:
        ret = os.system(cmd)
        if ret:
            return f'copy_dir_from_node(): {cmd} returned {ret}'
    except Exception as e:
        return f'copy_dir_from_node(): {cmd} failed with {e}'
    return ''

# create a directory on DAQ nodes
#
def make_remote_dirs(daq_config: dict[str, Any], dirname: str) -> None:
    """Create a directory on all configured remote DAQ nodes via SSH.

    Args:
        daq_config: Raw DAQ configuration dictionary.
        dirname: Path of the directory to create.

    Raises:
        Exception: If the SSH command fails on any node.
    """
    for node in daq_config['daq_nodes']:
        cmd = 'ssh {}@{} "cd {}; mkdir {}"'.format(
            node['username'], node['ip_addr'], node['data_dir'], dirname
        )
        print(cmd)
        ret = os.system(cmd)
        if ret:
            raise Exception(f'{cmd} returned {ret}')

# copy config files to run dirs on DAQ nodes
#
def copy_config_files(daq_config: dict[str, Any], run_dir: str, verbose: bool = False) -> None:
    """Distribute all observatory configuration files to remote DAQ nodes.

    Args:
        daq_config: Raw DAQ configuration dictionary.
        run_dir: Name of the target run directory on the remote nodes.
        verbose: If True, prints transfer details.
    """
    for node in daq_config['daq_nodes']:
        for f in config_file.config_file_names:
            copy_file_to_node(f, daq_config, node, run_dir, verbose)

# copy hashpipe binary and scripts to data dirs on DAQ nodes
#
def copy_daq_files(daq_config: DaqConfigValidator | dict[str, Any]) -> None:
    """Bootstrap remote DAQ nodes with essential software and scripts.
    
    Copies the Hashpipe binary (if present) and support scripts required 
    for remote data acquisition.

    Args:
        daq_config: Validated DAQ configuration model or dict.
    """
    if isinstance(daq_config, dict):
        daq_config = DaqConfigValidator(**daq_config)
    
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
        node_dict = node.model_dump()
        if hashpipe_so_exist:
            copy_file_to_node(hashpipe_so, daq_config.model_dump(), node_dict)
        copy_file_to_node('daq_scripts/start_daq.py', daq_config.model_dump(), node_dict)
        copy_file_to_node('daq_scripts/stop_daq.py', daq_config.model_dump(), node_dict)
        copy_file_to_node('daq_scripts/status_daq.py', daq_config.model_dump(), node_dict)
        copy_file_to_node('utils/util.py', daq_config.model_dump(), node_dict)
        copy_file_to_node('utils/pff.py', daq_config.model_dump(), node_dict)
        copy_file_to_node('daq_scripts/video_daq.py', daq_config.model_dump(), node_dict)

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
