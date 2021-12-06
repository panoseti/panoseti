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

# link modules to DAQ nodes:
# - in the daq_config data structure, add a list "modules"
#   to each daq node object, of the module objects
#   in the quabo_uids data structure;
# - in the quabo_ids data structure, in each module object,
#   add a link "daq_node" to the DAQ node that's handling it.
#
def associate(daq_config, quabo_uids):
    for n in daq_config['daq_nodes']:
        n['modules'] = []
    for dome in quabo_ids['domes']:
        for module in dome['modules']:
            daq_node = config_file.module_num_to_daq_node(daq_config, module['num'])
            daq_node['modules'].append(module)
            module['daq_node'] = daq_node


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

    if (nops == 0):
        raise Exception('no op specified')
    if (nops > 1):
        raise Exception('must specify a single op')
    if (nsel > 1):
        raise Exception('only one selector allowed')


    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids();

    associate(obs_config, daq_config, quabo_uids)
