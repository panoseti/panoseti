#! /usr/bin/env python

# copy files to/from daq nodes
#
# options when run as a script:
# --config              copy data products config file to noes
# --hashpipe            copy hashpipe executable (HSD_hashpipe.so) to nodes
# --get_data run_dir    copy data files in given run dir from daq nodes

import config_file, sys, os
import util

# copy a file to a node
#
def copy_file_to_node(file, node, run_dir=''):
    dest_path = node['dir']
    if run_dir:
        dest_path += '/%s'%(run_dir)
    cmd = 'scp %s %s@%s:%s'%(
        file, node['username'], node['ip_addr'], dest_path
    )
    print(cmd)
#os.system(cmd)

# copy the contents of a run dir from a DAQ node.
# to the corresponding run dir on this node
# scp doesn't let you do this directly,
# so we copy the dir to a temp directory (data/IP_ADDR),
# then move (rename) the files into the target dir
#
def copy_dir_from_node(data_dir, run_name, node):
    run_dir_path = '%s/%s'%(data_dir, run_name)
    if not run_dir_path:
        raise Exception('No run dir %s'%run_name)

    # make a temp dir if needed
    #
    tmp_dir = '%s/%s'%(data_dir, node['ip_addr'])
    print(tmp_dir)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    cmd = 'scp -r %s@%s:%s/%s %s'%(
        node['username'], node['ip_addr'], node['dir'], run_name, tmp_dir
    )
    print(cmd)
#os.system(cmd)

    cmd = 'mv %s/* %s'%(tmp_dir, run_dir_path)
    print(cmd)
#os.system(cmd)

#os.rmdir(tmp_dir)

# create a directory on DAQ nodes
#
def make_remote_dirs(daq_config, dirname):
    for node in daq_config['daq_nodes']:
        cmd = 'ssh %s@%s "cd %s; mkdir %s"'%(
            node['username'], node['ip_addr'], node['dir'], dirname
        )
        print(cmd)
#os.system(cmd)

# copy config files to a run dir on a DAQ node
#
def copy_config_files(node, run_dir):
    copy_file_to_node('daq_config.json', node, run_dir)
    copy_file_to_node('obs_config.json', node, run_dir)

def copy_hashpipe(daq_config):
    for node in daq_config['daq_nodes']:
        copy_file_to_node('../daq/HSD_hashpipe.so', node)

if __name__ == "__main__":

    def usage():
        print('''options:
    --config: copy config files: obs_config.json, data_config.json
    --hashpipe: copy hashpipe .so file
    ''')
        sys.exit()

    do_config = False
    do_hashpipe = False
    argv = sys.argv
    i = 1
    while i < len(argv):
        if argv[i] == '--config':
            do_config = True
        elif argv[i] == '--hashpipe':
            do_hashpipe = True
        else:
            usage()
        i += 1

    if not do_config and not do_hashpipe:
        usage()

    c = config_file.get_daq_config()
    if do_config:
        copy_config(daq_nodes)
    if do_hashpipe:
        copy_hashpipe(daq_nodes)
