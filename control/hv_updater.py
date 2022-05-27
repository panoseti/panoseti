#! /usr/bin/env python3

# Periodically updates the high-voltage values in each quabo
# based on their latest temperature in the Redis database.
# See https://github.com/panoseti/panoseti/issues/47.

import time
import redis
import json

import redis_utils
import quabo_driver
import config_file
import util

# Seconds between updates.
UPDATE_INTERVAL = 10
# Get quabo and detector info.
quabo_info = config_file.get_quabo_info()
detector_info = config_file.get_detector_info()
quabo_uids = config_file.get_quabo_uids()
obs_config = config_file.get_obs_config()
# Store the relation between quabo uids and their 'QUABO_*' keys in Redis.
quabo_uids_and_rkeys = {}


def get_adjusted_detector_hv(det_serial_num: str, temp: float) -> float:
    """Given a detector serial number, returns the appropriate
    temperature-adjusted high-voltage value. TEMP is assumed
    to be a temperature in degrees Celsius."""
    nominal_hv = detector_info[det_serial_num]
    # Formula from GitHub Issue 47.
    return nominal_hv + (temp - 25) * 0.054


def update_quabo_hv(quabo: quabo_driver.QUABO,
                 det_serial_nums: list,
                 temp: float):
    """Helper method for the function update_all. Updates the high-voltage
    values in the quabo."""
    for detector_index in range(4):
        det_serial_num = det_serial_nums[detector_index]
        adjusted_hv = get_adjusted_detector_hv(det_serial_num, temp)
        quabo.hv_set_chan(detector_index, adjusted_hv)


def update_all_quabo_hv(r: redis.Redis):
    """Iterate through each quabo in the observatory and update its detectors'
    high-voltages through a call to update_quabo."""
    for dome in obs_config["domes"]:
        for module in dome["modules"]:
            module_ip_addr = module['ip_addr']
            for quabo_index in range(4):
                # Get quabo uid
                uid = util.quabo_uid(module, quabo_uids, quabo_index)
                if uid == '':
                    continue

                # Get quabo object
                q_ip_addr = util.quabo_ip_addr(module_ip_addr, quabo_index)
                quabo_obj = quabo_driver.QUABO(q_ip_addr)

                # Get the list of detector serial numbers for this quabo.
                q_info = quabo_info[uid]
                detector_serial_nums = q_info["detector_serialno"]

                # Note: currently the key 'TEMP1' does not exist in the HK Redis,
                # so line 78 will throw an AttributeError.

                # Get the temperature for this quabo from Redis.
                quabo_redis_key = quabo_uids_and_rkeys[uid]
                temp = r.hget(quabo_redis_key, 'TEMP1')  # TODO: fix capture_hk.py to save temperature HK data.
                temp = float(temp.decode("utf-8"))

                # Call helper to adjust high-voltages in this quabo.
                update_quabo_hv(quabo_obj, detector_serial_nums, temp)


def main():
    try:
        r = redis_utils.redis_init()
        # Populate quabo_uids_and_rkeys.
        for quabo_key in r.keys('QUABO_*'):
            uid = r.hget(quabo_key, 'UID').decode("utf-8")
            # Remove '0x' prefix from Redis UID to match UIDs in config files.
            uid = hex(int(uid, 16))[2:]
            quabo_uids_and_rkeys[uid] = quabo_key
        while True:
            update_all_quabo_hv(r)
            time.sleep(UPDATE_INTERVAL)
    except redis.RedisError as err:
        print("Redis error {0}".format(err))
        raise
    except:
        raise


main()
