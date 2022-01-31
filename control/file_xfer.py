#! /usr/bin/env python3

# copy files to/from daq nodes
#
# options when run as a script:
# --config              copy data products config file to noes
# --hashpipe            copy hashpipe executable (HSD_hashpipe.so) to nodes
# --get_data run_dir    copy data files in given run dir from daq nodes

import sys, os
import util, config_file

# copy a file to a DAQ node
#
def copy_file_to_node(file, daq_config, node, run_dir=''):
    dest_path = node['data_dir']
    if run_dir:
        dest_path += '/%s'%(run_dir)
    else:
        dest_path += '/'
    cmd = 'scp -q %s %s@%s:%s'%(
        file, node['username'], node['ip_addr'], dest_path
    )
    print(cmd)
    ret = os.system(cmd)
    if ret: raise Exception('%s returned %d'%(cmd, ret))

# copy the contents of a run dir from a DAQ node.
# to the corresponding run dir on this (head) node
# scp doesn't let you do this directly,
# so we copy the dir to a temp directory (data/IP_ADDR/run),
# then move (rename) the files into the target dir
#
def copy_dir_from_node(run_name, daq_config, node):
    local_data_dir = daq_config['head_node_data_dir']
    run_dir_path = '%s/%s'%(local_data_dir, run_name)
    if not run_dir_path:
        raise Exception('No run dir %s'%run_name)
    # run dir should have already been created, but just in case
    if not os.path.isdir(run_dir_path):
        os.mkdir(run_dir_path)

    # make a temp dir if needed
    #
    node_tmp_dir = '%s/%s'%(local_data_dir, node['ip_addr'])
    if not os.path.isdir(node_tmp_dir):
        os.mkdir(node_tmp_dir)

    # copy run dir from remote node to temp dir
    cmd = 'scp -q -r %s@%s:%s/%s %s'%(
        node['username'], node['ip_addr'],
        node['data_dir'], run_name,
        node_tmp_dir
    )
    print(cmd)
    ret = os.system(cmd)
    if ret: raise Exception('%s returned %d'%(cmd, ret))

    # move non-config files from temp dir to head node data dir
    run_tmp_dir = '%s/%s'%(node_tmp_dir, run_name)
    for fn in os.listdir(run_tmp_dir):
        if fn.find('config')>=0 or fn.find('quabo_uids')>=0:
            #os.unlink('%s/%s'%(tmp_dir, fn))
            continue
        cmd = 'mv %s/%s %s/'%(run_tmp_dir, fn, run_dir_path)
        print(cmd)
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

#os.rmdir(tmp_dir)

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
def copy_config_files(daq_config, run_dir):
    for node in daq_config['daq_nodes']:
        for f in config_file.config_file_names:
            copy_file_to_node(f, daq_config, node, run_dir)

# copy hashpipe binary and scripts to data dirs on DAQ nodes
#
def copy_hashpipe(daq_config):
    for node in daq_config['daq_nodes']:
        copy_file_to_node('../daq/HSD_hashpipe.so', daq_config, node)
        copy_file_to_node('start_daq.py', daq_config, node)
        copy_file_to_node('stop_daq.py', daq_config, node)
        copy_file_to_node('status_daq.py', daq_config, node)
        copy_file_to_node('util.py', daq_config, node)

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
