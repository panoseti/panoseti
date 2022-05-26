#! /usr/bin/env python3

# Periodically updates the high-voltage values in each quabo
# based on their latest temperature in the Redis database.
# See https://github.com/panoseti/panoseti/issues/47

import time
import redis
import json

import redis_utils
import quabo_driver
import config_file
import util

# Seconds between updates.
UPDATE_INTERVAL = 10
# Nominal operating high voltage at 25 degrees C for each detector array.
HV25CS= [0,0,0,0]  # TODO: Lookup actual values specified by Hamamatsu


def get_adjusted_hv(chan, temp: float):
    """Returns the adjusted high-voltage for a given detector, indexed by CHAN.
    Assumes TEMP is in degrees Celsius."""
    return HV25CS[chan] + (temp - 25) * 0.054

def get_module_ip_addrs():
    """Returns the IP address corresponding to each module."""
    obs_config = config_file.get_obs_config()
    obs_config[]

def update_quabo(quabo: quabo_driver.QUABO, temp: float):
    """Update the high-voltage values in the quabo QUABO."""

def update_module():
    """Update the high-voltage values in the module MODULE."""



def update_all_modules(r: redis.Redis):
    for quabo_name in r.keys('QUABO_*'):
        try:
            temp = r.hget(quabo_name, 'TEMP1')
            temp = float(temp.decode("utf-8"))
            # Get module IP address
            base_ip_addr =
            # Adjust voltage in each quabo
            for i in range(4):
                quabo_ip_addr = util.quabo_ip_addr(base_ip_addr, i)
                quabo = quabo_driver.QUABO(quabo_ip_addr)
                adjusted_hv = get_adjusted_hv(i, temp)
                quabo.hv_set_chan(i, adjusted_hv)
        except:
            # Note: the subprocess running this script (see util.py)
            # might cause this msg to be always hidden from a user:
            print("Failed to update %s".format(quabo_name))
            pass


def main():
    try:
        r = redis_utils.redis_init()
        obs_config = config_file.get_obs_config()
        while True:
            update_all_quabos(r)
            time.sleep(UPDATE_INTERVAL)
    except:
        raise


main()
