#! /usr/bin/env python3

# copy files to/from daq nodes
#
# options when run as a script:
# --config              copy data products config file to nodes
# --hashpipe            copy hashpipe executable (hashpipe.so) to nodes
# --get_data run_dir    copy data files in given run dir from daq nodes

import sys, os
import util
from glob import glob
sys.path.insert(0, '../util')
import config_file

# copy a file to a DAQ node
#
def copy_file_to_node(file, daq_config, node, run_dir='', verbose=False):
    dest_path = node['data_dir']
    if run_dir:
        dest_path += '/%s'%(run_dir)
    else:
        dest_path += '/'
    files = glob(file)
    for f in files:
        cmd = 'scp -q %s %s@%s:%s'%(
            f, node['username'], node['ip_addr'], dest_path
        )
        if verbose:
            print(cmd)
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

# Copy the contents of a module/run dir from a DAQ node
# to the corresponding run dir on this (head) node.
# scp doesn't let you do this directly,
# so we copy the dir to a temp directory (data/IP_ADDR/run),
# then move (rename) the files into the target dir
#
# return error message, or '' on success
#
def copy_dir_from_node(run_name, daq_config, node, module_id, verbose=False):
    local_data_dir = daq_config['head_node_data_dir']
    run_dir_path = '%s/%s'%(local_data_dir, run_name)

    if not os.path.isdir(run_dir_path):
        return 'copy_dir_from_node(): no run dir %s'%run_dir_path
        
    # copy stdout from remote node to this node
    cmd = 'rsync -P %s@%s:%s/%s/%s* %s'%(
        node['username'], node['ip_addr'],
        node['data_dir'], run_name, util.hp_stdout_prefix,
        run_dir_path
    )
    if verbose:
        print(cmd)
    try:
        ret = os.system(cmd)
    except:
        return 'copy_dir_from_node(): %s returned %d'%(cmd, ret)
    
    # copy process snapshot from remote node to this node
    cmd = 'rsync -P %s@%s:%s/%s/%s* %s'%(
        node['username'], node['ip_addr'],
        node['data_dir'], run_name, util.pss_prefix,
        run_dir_path
    )
    if verbose:
        print(cmd)
    try:
        ret = os.system(cmd)
    except:
        return 'copy_dir_from_node(): %s returned %d'%(cmd, ret)
    # copy PFF files from remote node to this node
    cmd = 'rsync -P %s@%s:%s/module_%d/%s/* %s'%(
        node['username'], node['ip_addr'],
        node['data_dir'], module_id, run_name,
        run_dir_path
    )
    if verbose:
        print(cmd)
    try:
        ret = os.system(cmd)
    except:
        return 'copy_dir_from_node(): %s returned %d'%(cmd, ret)
    return ''

# create a directory on DAQ nodes
#
def make_remote_dirs(daq_config, dirname):
    for node in daq_config['daq_nodes']:
        cmd = 'ssh %s@%s "cd %s; mkdir %s"'%(
            node['username'], node['ip_addr'], node['data_dir'], dirname
        )
        print(cmd)
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

# copy config files to run dirs on DAQ nodes
#
def copy_config_files(daq_config, run_dir, verbose=False):
    for node in daq_config['daq_nodes']:
        for f in config_file.config_file_names:
            copy_file_to_node(f, daq_config, node, run_dir, verbose)

# copy hashpipe binary and scripts to data dirs on DAQ nodes
#
def copy_daq_files(daq_config):
    for node in daq_config['daq_nodes']:
        copy_file_to_node('../daq/hashpipe.so', daq_config, node)
        copy_file_to_node('start_daq.py', daq_config, node)
        copy_file_to_node('stop_daq.py', daq_config, node)
        copy_file_to_node('status_daq.py', daq_config, node)
        copy_file_to_node('util.py', daq_config, node)
        copy_file_to_node('../util/pff.py', daq_config, node)
        copy_file_to_node('video_daq.py', daq_config, node)

if __name__ == "__main__":

    def usage():
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
        copy_hashpipe(daq_config)
