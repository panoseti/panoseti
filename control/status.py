#! /usr/bin/env python3

# show the status of a recording run

import subprocess
import util, config_file

def status():
    run_name = util.read_run_name()
    if not run_name:
        print("No run is in progress")
        return
    print('Run in progress: %s'%run_name)
    # in theory should use config files in run dir
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    data_config = config_file.get_data_config()
    util.associate(daq_config, quabo_uids)
    my_ip = util.local_ip()
    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        ip_addr = node['ip_addr']
        username = node['username']
        if ip_addr == my_ip:
            print('local status')
        else:
            print('DAQ node %s:'%ip_addr)
            x = subprocess.run(['ssh',
                '%s@%s'%(username, ip_addr),
                'status_daq.py'],
                stdout = subprocess.PIPE
            )
            print(x.stdout)

status()
