#! /usr/bin/env python

# start a recording run:
# - figure out association of quabos and DAQ nodes,
#   based on config files
# - start the flow of data: set DAQ mode and dest IP addr of quabos
# - send commands to DAQ nodes to start hashpipe program

# based on matlab/startmodules.m, startqNph.m, changepeq.m
# options:
# --no_record       don't start hashpipe program on DAQ nodes (for debugging)

import os, sys
import copy_to_daq_nodes, util, config_file

# start data flow from the quabos
# for each one:
# - tell it where to send data
# - set its DAQ mode
#
def start_data_flow(quabo_uids, data_config):
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            dn = module['daq_node']
            for i in range(4):
                quabo = module['quabos'][i]
                if quabo['uid'] == '':
                    continue
# to be continued

# for each DAQ node that is getting data:
# - create run directory
# - copy config files to run directory
# - start hashpipe program
#
def start_recording(daq_config, data_config, run_name, no_record):
    for node in daq_config['daq_nodes']:
        if len(node['modules']) > 0:
            copy_to_daq_nodes.copy_config_files(node, run_name)
    if not no_record:
        for node in daq_config['daq_nodes']:
            if len(node['modules']) > 0:
                cmd = 'ssh %s@%s "cd %s; start_hashpipe.py"'%(
                    node['username'], node['ip_addr'], node['dir']
                )
                print(cmd)
                #os.system(cmd)

def start_run(obs_config, daq_config, data_config, quabo_ids, no_record=False):
    run_name = util.make_run_name(obs_config['name'], data_config['run_type'])
    data_dir = util.data_dir()
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    os.mkdir('%s/%s'%(data_dir, run_name))
    util.associate(daq_config, quabo_uids)
    start_data_flow(quabo_uids, data_config)
    util.start_hk_recorder(run_name)
    start_recording(daq_config, data_config, run_name, no_record)
    util.write_run_name(run_name)

if __name__ == "__main__":
    argv = sys.argv
    i = 1
    no_record = False
    while i < len(argv):
        if argv[i] == '--no_record':
            no_record = True
        else:
            raise Exception('bad arg %s'%argv[i])
        i += 1

    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    start_run(obs_config, daq_config, data_config, quabo_uids, no_record)
