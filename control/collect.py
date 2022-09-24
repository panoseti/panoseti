#! /usr/bin/env python3

# collect files from remote DAQ nodes at the end of a recording run
#
# --run_dir X    specify run dir

import os, sys
import file_xfer, util
sys.path.insert(0, '../util')
import config_file

def collect_data(daq_config, run_dir, verbose=False):
    my_ip = util.local_ip()
    for node in daq_config['daq_nodes']:
        for module in node['modules']:
            module_id = module['id']
            if node['ip_addr'] == my_ip:
                # head node is also a DAQ node.
                # Move files locally; if different volume, this will copy
                cmd = 'mv %s/module_%d/%s/* %s/%s'%(
                    node['data_dir'], module_id, run_dir,
                    daq_config['head_node_data_dir'], run_dir
                )
                if verbose:
                    print(cmd)
                ret = os.system(cmd)
                if ret:
                    raise Exception('command %s failed: %d'%(cmd, ret))
            else:
                file_xfer.copy_dir_from_node(
                    run_dir, daq_config, node, module_id, verbose
                )

# remove stuff from DAQ nodes no longer needed after run
# remote:
#    data/run
#    data/module_n/run
# local
#    data/module_n/run (should be empty dir)

def cleanup_daq(daq_config, run_dir, verbose=False):
    my_ip = util.local_ip()
    for node in daq_config['daq_nodes']:
        if node['ip_addr'] == my_ip:
            cmd = 'rm -rf %s/module_*/%s'%(
                node['data_dir'], run_dir
            )
            if verbose:
                print(cmd)
            ret = os.system(cmd)
            if ret:
                raise Exception('command %s failed: %d'%(cmd, ret))
        else:
            rcmd = 'rm -rf %s/module_*/%s; rm -rf %s/%s'%(
                node['data_dir'], run_dir,
                node['data_dir'], run_dir
            )
            cmd = 'ssh %s@%s "%s"'%(
                node['username'], node['ip_addr'], rcmd
            )
            if verbose:
                print(cmd)
            ret = os.system(cmd)
            if ret:
                raise Exception('command %s failed: %d'%(cmd, ret))

if __name__ == "__main__":
    i = 1
    run_dir = ''
    verbose = False
    while i<len(sys.argv):
        if sys.argv[i] == '--run_dir':
            i += 1
            run_dir = sys.argv[i]
        elif sys.argv[i] == '--verbose':
            verbose = True
        i += 1
    if not run_dir:
        run_dir = util.read_run_name()
        if not run_dir:
            print("No run found")
            sys.exit()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    config_file.associate(daq_config, quabo_uids)
    collect_data(daq_config, run_dir, verbose)
