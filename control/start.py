#! /usr/bin/env python3

# start a recording run:
#
# - figure out association of quabos and DAQ nodes,
#   based on config files
# - start the flow of data: set DAQ mode and dest IP addr of quabos
# - send commands to DAQ nodes to start hashpipe program
#
# fail if a recording run is in progress,
# or if recording activities are active

# based on matlab/startmodules.m, startqNph.m, changepeq.m

import os, sys, traceback, shutil
import config_file, util, file_xfer, quabo_driver

sys.path.insert(0, '../util')

import pff

# parse the data config file to get DAQ params for quabos
#
def get_daq_params(data_config):
    do_image = False
    image_usec = 1
    image_8bit = False
    do_ph = False
    bl_subtract = True
    if 'image' in data_config:
        do_image = True
        image = data_config['image']
        if 'integration_time_usec' not in image:
            raise Exception('missing integration_time_usec in data_config.json')
        if image['quabo_sample_size'] == 8:
            image_8bit = True
        elif image['quabo_sample_size'] != 16:
            raise Exception('quabo_sample_size must be 8 or 16')
        image_usec = image['integration_time_usec']
    if 'pulse_height' in data_config:
        do_ph = True
    daq_params = quabo_driver.DAQ_PARAMS(
        do_image, image_usec - 1, image_8bit, do_ph, bl_subtract
    )
    if 'flash_params' in data_config:
        fp = data_config['flash_params']
        daq_params.set_flash_params(fp['rate'], fp['level'], fp['width'])
    return daq_params

# Start data flow from the quabos.
# For each quabo:
# - tell it where to send HK packets
# - tell it where to send data packets
# - set its DAQ mode
#
def start_data_flow(quabo_uids, data_config, daq_config):
    daq_params = get_daq_params(data_config)        
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            if 'daq_node' not in module:
                continue
            base_ip_addr = module['ip_addr']
            module_id = util.ip_addr_to_module_id(base_ip_addr)
            daq_node = config_file.module_id_to_daq_node(daq_config, module_id)
            daq_node_ip_addr = daq_node['ip_addr']
            head_node_ip_addr = daq_config['head_node_ip_addr']
            for i in range(4):
                quabo = module['quabos'][i]
                if quabo['uid'] == '':
                    continue
                ip_addr = util.quabo_ip_addr(base_ip_addr, i)
                quabo = quabo_driver.QUABO(ip_addr)
                print('setting HK packet dest to %s on quabo %s'%(
                    head_node_ip_addr, ip_addr
                ))
                quabo.hk_packet_destination(daq_node_ip_addr)
                print('setting data packet dest to %s on quabo %s'%(
                    daq_node_ip_addr, ip_addr
                ))
                quabo.data_packet_destination(daq_node_ip_addr)
                print('setting DAQ mode on quabo %s'%ip_addr)
                quabo.send_daq_params(daq_params)
                quabo.close()

# start recording data
#   copy config files to run dir on head node
#   for each DAQ node that is getting data
#       create run directory
#       copy config files to run directory
#       start hashpipe program
#
def start_recording(data_config, daq_config, run_name, no_hv):
    my_ip = util.local_ip()

    # copy config files to run dir on this node
    local_data_dir = daq_config['head_node_data_dir']
    for f in config_file.config_file_names:
        shutil.copyfile(f, '%s/%s'%(local_data_dir, f))

    # make run directories on remote DAQ nodes
    #
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        ip_addr = node['ip_addr']
        if ip_addr == my_ip:
            continue
        username = node['username']
        data_dir = node['data_dir']
        cmd = 'ssh %s@%s "mkdir %s/%s"'%(
            username, ip_addr, data_dir, run_name
        )
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

    # copy config files to DAQ nodes
    file_xfer.copy_config_files(daq_config, run_name)

    # start recording HK data
    util.start_hk_recorder(daq_config, run_name)

    if not no_hv:
        # start high-voltage updater
        util.start_hv_updater()

        # start module temperature monitor
        util.start_module_temp_monitor()

    # start hashpipe on DAQ nodes

    if 'max_file_size_mb' in data_config.keys():
        max_file_size_mb = int(data_config['max_file_size_mb'])
    else:
        max_file_size_mb = util.default_max_file_size_mb
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        username = node['username']
        data_dir = node['data_dir']
        remote_cmd = './start_daq.py --daq_ip_addr %s --run_dir %s --max_file_size_mb %d'%(
            node['ip_addr'], run_name, max_file_size_mb
        )
        if 'bindhost' in node.keys():
            remote_cmd += ' --bindhost %s'%node['bindhost']
        for m in node['modules']:
            module_id = util.ip_addr_to_module_id(m['ip_addr'])
            remote_cmd += ' --module_id %d'%module_id
        cmd = 'ssh %s@%s "cd %s; %s"'%(
            username, node['ip_addr'], data_dir, remote_cmd
        )
        print(cmd)
        ret = os.system(cmd)
        if ret: raise Exception('%s returned %d'%(cmd, ret))

def start_run(obs_config, daq_config, quabo_uids, data_config, no_hv):
    my_ip = util.local_ip()
    if my_ip != daq_config['head_node_ip_addr']:
        print('This is not the head node; see daq_config.json')
        return False

    rn = util.read_run_name()
    if (rn):
        print('A run is already in progress.  Run stop.py, then try again.')
        return False

    if util.is_hk_recorder_running():
        print('The HK recorder is running.  Run stop.py, then try again.')
        return False
        
    if not util.are_redis_daemons_running():
        print('Redis daemons are not running')
        util.show_redis_daemons()
        return False

    # if head node is also DAQ note, make sure data dirs are the same
    #
    for node in daq_config['daq_nodes']:
        if my_ip == node['ip_addr'] and daq_config['head_node_data_dir'] != node['data_dir']:
            print("Head node data dir doesn't match DAQ node data dir")
            return False

    try:
        run_name = pff.run_dir_name(obs_config['name'], data_config['run_type'])
        run_dir = '%s/%s'%(daq_config['head_node_data_dir'], run_name)
        os.mkdir(run_dir)
        config_file.associate(daq_config, quabo_uids)
        config_file.show_daq_assignments(quabo_uids)
        print('starting data flow from quabos')
        start_data_flow(quabo_uids, data_config, daq_config)
        print('starting recording')
        start_recording(data_config, daq_config, run_name, no_hv)
    except:
        print(traceback.format_exc())
        print("Couldn't start run.  Run stop.py, then try again.")
        print('If other users might be using the telescope, check with them;')
        print('running stop.py will kill their run.')
        return False
    util.write_run_name(daq_config, run_name)
    print('started run %s'%run_name)
    return True

if __name__ == "__main__":
    argv = sys.argv
    no_hv = False
    i = 1
    while i < len(argv):
        if argv[i] == '--no_hv':
            no_hv = True
        else:
            raise Exception('bad arg %s'%argv[i])
        i += 1

    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    start_run(obs_config, daq_config, quabo_uids, data_config, no_hv)
