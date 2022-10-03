#! /usr/bin/env python3

# show the status of a recording run

import subprocess, sys
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
        username = node['username']
        print('status on DAQ node %s:'%ip_addr)
        x = subprocess.run(['ssh',
            '%s@%s'%(username, ip_addr),
            'cd %s; ./status_daq.py'%(node['data_dir']),
            ],
            stdout = subprocess.PIPE
        )
        print(x.stdout.decode())

status()
