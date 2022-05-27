#! /usr/bin/env python3

# Periodically updates the high-voltage values in each quabo
# based on their latest temperature in the Redis database.
# See https://github.com/panoseti/panoseti/issues/47.

import time
import redis

import redis_utils
import quabo_driver
from config_file import get_detector_info, get_quabo_info

# Seconds between updates.
UPDATE_INTERVAL = 10
# Get quabo and detector info.
quabo_info = get_quabo_info()
detector_info = get_detector_info()


def get_adjusted_detector_hv(quabo_uid, detector_index, temp) -> float:
    """Given a detector specified by a QUABO_UID and a DETECTOR_INDEX in
     the range 0 to 3, returns a hv value adjusted for the given TEMP.
      Temp is assumed to be a temperature in degrees Celsius."""
    quabo = quabo_info[quabo_uid]
    detector_serialno = quabo['detector_serialno'][detector_index]
    nominal_hv = detector_info[str(detector_serialno)]
    # Formula from Github Issue 47.
    return nominal_hv + (temp - 25) * 0.054


def quabo_uid_to_module_ip_addr(quabo_uid):
    """Returns a module's ip address given one of its quabo's QUABO_UIDs."""

    return


def update_all(r: redis.Redis):
    """Update the high-voltage values in each quabo."""
    for quabo_name in r.keys('QUABO_*'):
        try:
            # Get quabo uid
            quabo_uid = r.hget(quabo_name, 'UID')
            # Get quabo IP address and create corresponding Quabo object
            quabo_ip_addr = quabo_uid_to_module_ip_addr(quabo_uid)
            quabo = quabo_driver.QUABO(quabo_ip_addr)
            # Get quabo temperature from Redis
            temp = r.hget(quabo_name, 'TEMP1') # TODO: save TEMP1 data in capture_hk.py
            temp = float(temp.decode("utf-8"))
            # Set voltage
            for det_index in range(4):
                adjusted_hv = get_adjusted_detector_hv(quabo_uid, det_index, temp)
                quabo.hv_set_chan(det_index, adjusted_hv)
        except:
            print("Failed to update the voltage of %s".format(quabo_name))
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
