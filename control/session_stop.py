#! /usr/bin/env python3

import sys

import power, util

sys.path.insert(0, '../util')
import config_file

def close_domes(obs_config):
    print('Close the shutters of these domes:')
    for dome in obs_config['domes']:
        print('   ', dome['name'])

def session_stop(obs_config):
    close_domes(obs_config)
    power.do_all(obs_config, 'off')
    util.stop_redis_daemons()

if __name__ == "__main__":
    obs_config = config_file.get_obs_config()
    session_stop(obs_config)