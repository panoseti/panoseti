#! /usr/bin/env python3

# Periodically updates the high-voltage values in each quabo
# based on their temperature in the Redis database.
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


def update_quabo(quabo: quabo_driver.QUABO, temp: float):
    """Update the high-voltage values in the quabo QUABO."""

def update_quabo(r: redis.Redis, quabo: quabo_driver.QUABO):
    """Update the high-voltage values in the quabo QUABO."""
    temp = r.hget(quabo_name, 'TEMP1')
    temp = float(temp.decode("utf-8"))
    try:
        det_serial_num = quabo_name[5:]
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


def update_all(r: redis.Redis, quabo_info, quabo_uids):
    for dome in quabo_uids["domes"]:
        for module in dome["modules"]:
            base_ip_addr = module['ip_addr']
            for i in range(4):
                qi_ip_addr = util.quabo_ip_addr(base_ip_addr, i)
                uid = util.quabo_uid(module, quabo_uids, i)
                if uid == '': continue
                qi_info = quabo_info[uid]
                qi_serial = qi_info["detector_serial"]
                qi = quabo_driver.QUABO(qi_ip_addr)
                update_quabo(r, qi)


def main():
    try:
        quabo_info = config_file.get_quabo_info()
        quabo_uids = config_file.get_quabo_uids()
        r = redis_utils.redis_init()
        while True:
            update_all(r, quabo_info, quabo_uids)
            time.sleep(UPDATE_INTERVAL)
    except:
        raise


main()
