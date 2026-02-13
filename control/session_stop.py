#! /usr/bin/env python3

import sys, os

import power
from utils import config_file, util

def session_stop(obs_config):
    power.do_all(obs_config, 'off')
    try:
        util.stop_redis_daemons()
    except PermissionError as perr:
        print("You don't have permission to stop the redis daemons. "
              "Try running 'sudo ./config.py --stop_redis_daemons'.")

if __name__ == "__main__":
    obs_config = config_file.get_obs_config()
    session_stop(obs_config)
    

