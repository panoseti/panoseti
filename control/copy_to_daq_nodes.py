#! /usr/bin/env python

# copy files to daq nodes
#
# --config      copy data products config file
# --hashpipe    copy hashpipe executable (HSD_hashpipe.so)

import config_file, sys, os


# copy a file to a node
#
def copy(file, node):
    cmd = 'scp %s %s@%s:%s'%(file, node['username'], node['ip_addr'], node['dir'])
    print(cmd)
    os.system(cmd)

# create a directory on DAQ nodes
#
def make_remote_dirs(daq_config, dirname):
    for node in daq_config['daq_nodes']:
        cmd = 'ssh %s@%s "cd %s; mkdir %s"'%(node['username'], node['ip_addr'], node['dir'], dirname)
        os.system(cmd)

# copy files to all DAQ nodes
#
def copy_all(daq_config, do_config=True, do_hashpipe=True):
    for node in daq_config['daq_nodes']:
        if do_config:
            copy('daq_config.json', node)
            copy('obs_config.json', node)
        if do_hashpipe:
            copy('../daq/HSD_hashpipe.so', node)

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
    copy_all(do_config, do_hashpipe)
