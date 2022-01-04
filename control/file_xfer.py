#! /usr/bin/env python3

# copy files to/from daq nodes
#
# options when run as a script:
# --config              copy data products config file to noes
# --hashpipe            copy hashpipe executable (HSD_hashpipe.so) to nodes
# --get_data run_dir    copy data files in given run dir from daq nodes

import config_file, sys, os
import util

# copy a file to a DAQ node
#
def copy_file_to_node(file, daq_config, node, run_dir=''):
    dest_path = daq_config['data_dir']
    if run_dir:
        dest_path += '/%s'%(run_dir)
    cmd = 'scp -q %s %s@%s:%s'%(
        file, daq_config['username'], node['ip_addr'], dest_path
    )
    print(cmd)
    ret = os.system(cmd)
    if ret: raise Exception('%s returned %d'%(cmd, ret))

# copy the contents of a run dir from a DAQ node.
# to the corresponding run dir on this node
# scp doesn't let you do this directly,
# so we copy the dir to a temp directory (data/IP_ADDR),
# then move (rename) the files into the target dir
#
def copy_dir_from_node(data_dir, run_name, daq_config, node):
    run_dir_path = '%s/%s'%(data_dir, run_name)
    if not run_dir_path:
        raise Exception('No run dir %s'%run_name)

    # make a temp dir if needed
    #
    tmp_dir = '%s/%s'%(data_dir, node['ip_addr'])
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    cmd = 'scp -q -r %s@%s:%s/%s %s'%(
        daq_config['username'], node['ip_addr'], daq_config['data_dir'], run_name, tmp_dir
    )
    print(cmd)
    ret = os.system(cmd)
    if ret: raise Exception('%s returned %d'%(cmd, ret))

    cmd = 'mv %s/* %s'%(tmp_dir, run_dir_path)
    print(cmd)
    ret = os.system(cmd)
    if ret: raise Exception('%s returned %d'%(cmd, ret))

#os.rmdir(tmp_dir)

# create a directory on DAQ nodes
#
def make_remote_dirs(daq_config, dirname):
    for node in daq_config['daq_nodes']:
        cmd = 'ssh %s@%s "cd %s; mkdir %s"'%(
            node['username'], node['ip_addr'], node['dir'], dirname
        )
        print(cmd)
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

# copy config files to run dirs on DAQ nodes
#
def copy_config_files(daq_config, run_dir):
    for node in daq_config['daq_nodes']:
        copy_file_to_node('data_config.json', daq_config, node, run_dir)
        copy_file_to_node('obs_config.json', daq_config, node, run_dir)
        copy_file_to_node('quabo_uids.json', daq_config, node, run_dir)
        copy_file_to_node('daq_config.json', daq_config, node, run_dir)

def copy_hashpipe(daq_config):
    for node in daq_config['daq_nodes']:
        copy_file_to_node('../daq/HSD_hashpipe.so', daq_config, node)
        copy_file_to_node('start_daq.py', daq_config, node)
        copy_file_to_node('stop_daq.py', daq_config, node)
        copy_file_to_node('record_time.py', daq_config, node)

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
