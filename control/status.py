#! /usr/bin/env python3

# show the status of a recording run

import subprocess, sys, json
import util
sys.path.insert(0, '../util')
import config_file

def status():
    run_name = util.read_run_name()
    if run_name:
        print('Run in progress: %s'%run_name)
    else:
        print("No run is in progress")

    if util.is_hk_recorder_running():
        print('HK recorder is running')
    else:
        print('HK recorder is not running')

    # in theory should use config files in run dir
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    config_file.associate(daq_config, quabo_uids)
    my_ip = util.local_ip()
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        ip_addr = node['ip_addr']
        print('status on DAQ node %s:'%ip_addr)
        j = util.get_daq_node_status(node)
        #print(j)
        if j['hashpipe_running']:
            print('   hashpipe is running')
        else:
            print('   hashpipe is not running')
        if 'current_run' in j.keys():
            print('   current run:', ['current_run'])
            if 'current_run_disk' in j.keys():
                print('   disk usage:', ['current_run_disk'])
            else:
                print("   run dir doesn't exist")
        else:
            print('   no current run')
            
        vols = j['vols']
        print('   volumes:')
        for name in vols.keys():
            vol = vols[name]
            print('      name:', name)
            print('         free space: %.2fGB'%(vol['free']/1e9))
            print('         modules:', vol['modules'])

status()
