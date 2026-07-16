#! /usr/bin/env python3

# collect files from remote DAQ nodes at the end of a recording run
#
# options when run as a cmdline script:
#
# --run_dir X   specify run dir
# --cleanup     clean up DAQ nodes; don't collect
# --verbose
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import file_xfer, util, config_file

# return '' if data collection was successful, else error msg
#
def collect_data(daq_config, run_dir, verbose=False):
    my_ip = util.local_ip()
    error_msg = ''
    for node in daq_config['daq_nodes']:
        for module in node['modules']:
            module_id = module['id']
            if node['ip_addr'] in my_ip:
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
                    error_msg += 'command %s failed: %d'%(cmd, ret)
            else:
                error_msg += file_xfer.copy_dir_from_node(
                    run_dir, daq_config, node, module_id, verbose
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
def cleanup_daq(daq_config, run_dir, verbose=False):
    my_ip = util.local_ip()
    error_msg = ''
    for node in daq_config['daq_nodes']:
        if node['ip_addr'] in my_ip:
            cmd = 'rm -rf %s/module_*/%s'%(
                node['data_dir'], run_dir
            )
            if verbose:
                print(cmd)
            ret = os.system(cmd)
            if ret:
                error_msg += 'cleanup_daq(): %s returned %d '%(cmd, ret)
        else:
            rcmd = 'rm -rf %s/module_*/%s; rm -rf %s/%s'%(
                node['data_dir'], run_dir,
                node['data_dir'], run_dir
            )
            if 'port_forwarding' in node:
                cmd = 'ssh -p %d %s@%s "%s"'%(
                    node['port_forwarding']['port'], node['username'], node['port_forwarding']['gw_ip'], rcmd
                )
            else:
                cmd = 'ssh %s@%s "%s"'%(
                    node['username'], node['ip_addr'], rcmd
                )
            if verbose:
                print(cmd)
            ret = os.system(cmd)
            if ret:
                error_msg += 'cleanup_daq(): %s returned %d '%(cmd, ret)
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
        run_dir = util.read_run_name()
        if not run_dir:
            raise Exception("No run found")
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    config_file.associate(daq_config, quabo_uids)
    if cleanup:
        cleanup_daq(daq_config, run_dir, verbose)
    else:
        ret = collect_data(daq_config, run_dir, verbose)
        print('success' if ret else 'failed')
