#! /usr/bin/env python3

"""
Script for capturing metadata from each ethernet outlet
and storing it in the Redis database.
"""
from datetime import datetime
import time

import power
import redis_utils
sys.path.insert(0, '../util')
import config_file

# Time between updates.
UPDATE_INTERVAL = 1


def get_wps_fields(wps_dict):
    """Creates a dictionary of values to write into Redis."""
    try:
        power_status = "ON" if power.quabo_power_query(wps_dict) else "OFF"
    except Exception:
        print(f'capture_power.py: Failed to query {wps_dict}. The login info for this UPS may be incorrect."')
        raise
    rkey_fields = {
        'Computer_UTC': time.time(),
        'POWER': power_status
    }
    return rkey_fields


def get_wps_rkey(wps_key):
    """Returns the Redis key for the wps named 'wps_key'."""
    return wps_key.upper()


def main():
    r = redis_utils.redis_init()
    obs_config = config_file.get_obs_config()
    wps_keys = [key for key in obs_config.keys() if 'wps' in key.lower()]
    print("capture_power.py: Running...")
    while True:
        for wps_key in wps_keys:
            rkey = get_wps_rkey(wps_key)
            wps_dict = obs_config[wps_key]
            fields = get_wps_fields(wps_dict)
            redis_utils.store_in_redis(r, rkey, fields)
        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    main()

