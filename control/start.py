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

import os, sys, shutil
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

# start recording data
#   copy config files to run dir on head node
#   for each DAQ node that is getting data
#       create run directory
#       copy config files to run directory
#       start hashpipe program
#
def start_recording(data_config, daq_config, run_name):
    username = daq_config['daq_node_username']
    data_dir = daq_config['daq_node_data_dir']
    my_ip = util.local_ip()

    # copy config files to run dir on this node
    local_data_dir = daq_config['head_node_data_dir']
    for f in util.config_file_names:
        shutil.copyfile(f, '%s/%s'%(local_data_dir, f))

    # make run directories on DAQ nodes
    #
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        ip_addr = node['ip_addr']
        if ip_addr == my_ip:
            continue
        cmd = 'ssh %s@%s "mkdir %s/%s"'%(
            username, ip_addr, data_dir, run_name
        )
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

    # copy config files to DAQ nodes
    file_xfer.copy_config_files(daq_config, run_name)

    # start recording HK data
    #util.start_hk_recorder(daq_config, run_name)

    # start hashpipe on DAQ nodes

    if 'max_file_size_mb' in data_config.keys():
        max_file_size_mb = int(data_config['max_file_size_mb'])
    else:
        max_file_size_mb = util.default_max_file_size_mb
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        remote_cmd = './start_daq.py --daq_ip_addr %s --run_dir %s --max_file_size_mb %d'%(
            node['ip_addr'], run_name, max_file_size_mb
        )
        for m in node['modules']:
            module_id = util.ip_addr_to_module_id(m['ip_addr'])
            remote_cmd += ' --module_id %d'%module_id
        cmd = 'ssh %s@%s "cd %s; %s"'%(
            username, node['ip_addr'], data_dir, remote_cmd
        )
        print(cmd)
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

def start(obs_config, daq_config, quabo_uids, data_config):
    run_name = pff.run_dir_name(obs_config['name'], data_config['run_type'])
    run_dir = '%s/%s'%(daq_config['head_node_data_dir'], run_name)
    os.mkdir(run_dir)
    util.write_run_name(run_name)
    util.associate(daq_config, quabo_uids)
    util.show_daq_assignments(quabo_uids)
    print('starting data flow from quabos')
    start_data_flow(quabo_uids, data_config)
    print('starting recording')
    start_recording(data_config, daq_config, run_name)

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
