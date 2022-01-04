#! /usr/bin/env python3

# start or stop recording:
# - figure out association of quabos and DAQ nodes,
#   based on config files
# - start the flow of data: set DAQ mode and dest IP addr of quabos
# - send commands to DAQ nodes to start hashpipe program

# based on matlab/startmodules.m, startqNph.m, changepeq.m
# options:
# --start       start recording data
# --stop        stop recording data

import os, sys
import config_file
import util, file_xfer, quabo_driver, pff

# parse the data config file to get DAQ params for quabos
#
def get_daq_params(data_config):
    do_image = False;
    do_ph = False;
    image_usec = 0
    bl_subtract = True
    if 'image' in data_config:
        image = data_config['image']
        if 'integration_time_usec' not in image:
            raise Exception('missing integration_time_usec in data_config.json')
        image_usec = image['integration_time_usec']
    if 'pulse_height' in data_config:
        do_ph = True
    return quabo_driver.DAQ_PARAMS(do_image, image_usec, do_ph, bl_subtract)

# start data flow from the quabos
# for each one:
# - tell it where to send data
# - set its DAQ mode
#
def start_data_flow(quabo_uids, data_config):
    daq_params = get_daq_params(data_config)        
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            if 'daq_node' not in module:
                continue
            base_ip_addr = module['ip_addr']
            for i in range(4):
                quabo = module['quabos'][i]
                if quabo['uid'] == '':
                    continue
                ip_addr = util.quabo_ip_addr(base_ip_addr, i)
                print('starting quabo %s'%ip_addr)
                quabo = quabo_driver.QUABO(ip_addr)
                quabo.send_daq_params(daq_params)
                quabo.close()

# for each DAQ node that is getting data:
# - create run directory
# - copy config files to run directory
# - start hashpipe program
#
def start_recording(daq_config, run_name):
    username = daq_config['username']
    data_dir = daq_config['data_dir']
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        cmd = 'ssh %s@%s "mkdir %s/%s"'%(
            username, node['ip_addr'], data_dir, run_name
        )
        print(cmd)
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

    file_xfer.copy_config_files(daq_config, run_name)

    # start hashpipe on DAQ nodes
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        cmd = 'ssh %s@%s "cd %s; ./start_daq.py %s"'%(
            username, node['ip_addr'], data_dir, run_name
        )
        print(cmd)
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

def start(obs_config, daq_config, quabo_uids, data_config):
    run_name = pff.run_dir_name(obs_config['name'], data_config['run_type'])
    util.write_run_name(run_name)
    util.associate(daq_config, quabo_uids)
    util.show_daq_assignments(quabo_uids)
    print('starting data flow from quabos')
    start_data_flow(quabo_uids, data_config)
    print('starting recording')
    start_recording(daq_config, run_name)

if __name__ == "__main__":
    argv = sys.argv
    nops = 0
    i = 1
    while i < len(argv):
        raise Exception('bad arg %s'%argv[i])
        i += 1

    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    start(obs_config, daq_config, quabo_uids, data_config)
