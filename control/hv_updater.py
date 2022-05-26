#! /usr/bin/env python3

# Periodically updates the high-voltage values in each quabo
# based on their latest temperature in the Redis database.
# See https://github.com/panoseti/panoseti/issues/47

import time
import redis

import redis_utils
import quabo_driver

# Seconds between updates.
UPDATE_INTERVAL = 10
# Nominal operating high voltage at 25 degrees C for each detector array.
HV25CS= [0,0,0,0]  # TODO: Lookup actual values specified by Hamamatsu


def get_adjusted_hv(chan, temp: float):
    """Returns the adjusted high-voltage for a given detector, indexed by CHAN.
    Assumes TEMP is in degrees Celsius."""
    return HV25CS[chan] + (temp - 25) * 0.054


def update_all(r: redis.Redis):
    """Update the high-voltage values in each quabo."""
    for quabo_name in r.keys('QUABO_*'):
        try:
            temp = r.hget(quabo_name, 'TEMP1')
            temp = float(temp.decode("utf-8"))
            # Get Quabo IP address and create corresponding Quabo object
            ip_addr = ... # TODO: figure out how to get the IP address from Quabo uid.
            quabo = quabo_driver.QUABO(ip_addr)
            # Set voltage
            for chan in range(4):
                adjusted_hv = get_adjusted_hv(chan, temp)
                quabo.hv_set_chan(chan, adjusted_hv)
        except:
            # Note: the subprocess running this script (see util.py)
            # might cause this msg to be always hidden from a user:
            print("Failed to update %s".format(quabo_name))
            pass


def main():
    try:
        r = redis_utils.redis_init()
        while True:
            update_all(r)
            time.sleep(UPDATE_INTERVAL)
    except:
        raise


main()
