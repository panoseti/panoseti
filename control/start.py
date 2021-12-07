#! /usr/bin/env python

# start or stop recording:
# - figure out association of quabos and DAQ nodes,
#   based on config files
# - start the flow of data: set DAQ mode and dest IP addr of quabos
# - send commands to DAQ nodes to start hashpipe program

# based on matlab/startmodules.m, startqNph.m, changepeq.m
# options:
# --start       start recording data
# --stop        stop recording data

import config_file, sys
import copy_to_daq_nodes

# link modules to DAQ nodes:
# - in the daq_config data structure, add a list "modules"
#   to each daq node object, of the module objects
#   in the quabo_uids data structure;
# - in the quabo_uids data structure, in each module object,
#   add a link "daq_node" to the DAQ node that's handling it.
#
def associate(daq_config, quabo_uids):
    for n in daq_config['daq_nodes']:
        n['modules'] = []
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            daq_node = config_file.module_num_to_daq_node(daq_config, module['num'])
            daq_node['modules'].append(module)
            module['daq_node'] = daq_node

# show which module is going to which data recorder
#
def show_daq_assignments(quabo_uids):
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            for i in range(4):
                q = module['quabos'][i];
                print("data from quabo %s (%s) -> DAQ node %s"
                    %(q['uid'], quabo_ip_address(q['ip_addr'], i))
                )

# start data flow from the quabos
# for each one:
# - tell it where to send data
# - set its DAQ mode
#
def start_data_flow(quabo_uids, data_config):
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            for i in range(4):
                quabo = module['quabos'][i]
                if quabo['uid'] == '':
                    continue
                dn = quabo['daq_node']

# for each DAQ node that is getting data:
# - create run directory
# - copy config files to run directory
# - start hashpipe program
#
def start_recording(daq_config, data_config):
    run_name = pff.run_dir_name(obs_config['name'], data_config['run_type'])
    copy_to_daq_nodes.copy_all(daq_config, run_name)
    start_hashpipe(daq_config, run_name)

if __name__ == "__main__":
    argv = sys.argv
    nops = 0
    i = 1
    while i < len(argv):
        if argv[i] == '--start':
            nops += 1
            op = 'start'
        elif argv[i] == '--stop':
            nops += 1
            op = 'stop'
        else:
            raise Exception('bad arg %s'%argv[i])
        i += 1

    if nops == 0:
        raise Exception('no op specified')
    if nops > 1:
        raise Exception('must specify a single op')


    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()

    associate(daq_config, quabo_uids)
    show_daq_assignments(quabo_uids)
    start_data_flow(quabo_uids)
    start_recording(daq_config)
