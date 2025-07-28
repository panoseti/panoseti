#! /usr/bin/env python3

import sys, os

import power
from tools import skymap_helper
from utils import config_file, util

def close_domes(obs_config):
    print('Close the shutters of these domes:')
    for dome in obs_config['domes']:
        print('   ', dome['name'])

def session_stop(obs_config):
    close_domes(obs_config)
    power.do_all(obs_config, 'off')
    try:
        util.stop_redis_daemons()
    except PermissionError as perr:
        print("You don't have permission to stop the redis daemons. "
              "Try running 'sudo ./config.py --stop_redis_daemons'.")

if __name__ == "__main__":
    try:
        os.remove('obs_comments.txt')
    except:
        pass
    obs_config = config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    # gen skymap_info.json first, then stop the session
    skymap_helper.stop_skymap_info_gen()
    # get run name, and copy the skymap_info.json to the run dir
    with open('tmp/skymap_info_dir') as f:
        skymap_info_dir = f.read().strip()
    run_dir = daq_config['head_node_data_dir'] + '/' + skymap_info_dir
    print(run_dir)
    skymap_helper.copy_skymap_info_to_run_dir(run_dir)
    session_stop(obs_config)
    

