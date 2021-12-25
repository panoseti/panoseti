#! /usr/bin/env python

# copy files to daq nodes
#
# --config      copy data products config file
# --hashpipe    copy hashpipe executable (HSD_hashpipe.so)

import config_file, sys, os


# copy a file to data dir on a node
#
def copy_to_data_dir(file, node, daq_config):
    username = daq_config['username']
    data_dir = daq_config['data_dir']
    cmd = 'scp %s %s@%s:%s'%(file, username, node['ip_addr'], data_dir)
    print(cmd)
    os.system(cmd)

# copy a file to a run dir
#
def copy_to_run_dir(file, node, run_dir, daq_config):
    username = daq_config['username']
    data_dir = daq_config['data_dir']
    cmd = 'scp %s %s@%s:%s/%s'%(file, username, node['ip_addr'], data_dir, run_dir)
    print(cmd)
    os.system(cmd)

# create a run directory on DAQ nodes
#
def make_remote_dirs(daq_config, run_dirname):
    username = daq_config['username']
    data_dir = daq_config['data_dir']
    for node in daq_config['daq_nodes']:
        cmd = 'ssh %s@%s "cd %s; mkdir %s"'%(username, node['ip_addr'], data_dir, run_dirname)
        print(cmd)
    os.system(cmd)

# copy hashpipe .so data dir on all DAQ nodes
#
def copy_hashpipe(daq_config):
    for node in daq_config['daq_nodes']:
        copy_to_data_dir('../daq/HSD_hashpipe.so', node, daq_config)

def copy_config_files(daq_config, run_dir):
    for node in daq_config['daq_nodes']:
        copy_to_run_dir('daq_config.json', node, run_dir, daq_config)
        copy_to_run_dir('obs_config.json', node, run_dir, daq_config)
        copy_to_run_dir('data_config.json', node, run_dir, daq_config)

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

    daq_config = config_file.get_daq_config()
    if do_hashpipe:
        copy_hashpipe(daq_config)
    if do_config:
        make_remote_dirs(daq_config, 'test_run_dir')
        copy_config_files(daq_config, 'test_run_dir')
