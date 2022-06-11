#! /usr/bin/env python3

"""
Script for periodically updating the high-voltage values in every detector
active in the observatory. The adjustments are based on real-time temperature
data collected by housekeeping programs and detector settings specified by
the detector manufacturer, Hamamatsu.
See https://github.com/panoseti/panoseti/issues/47 for info about the issue
this code resolves.
See the Hamamatsu datasheet for its MPPC arrays: S13361-3050 series
for more info about the detector constants used in this script.
"""

import time

import redis

import redis_utils
import quabo_driver
import config_file
import util

#-------------- CONSTANTS ---------------#

# Seconds between updates.
UPDATE_INTERVAL = 5

# Min & max detector operating temperatures (degrees Celsius).
MIN_TEMP = -20.0
MAX_TEMP = 60.0

#--------- Implementation Globals --------#

# Get quabo and detector info.
quabo_info = config_file.get_quabo_info()
detector_info = config_file.get_detector_info()
quabo_uids = config_file.get_quabo_uids()

# Dict inverting the relation between a quabo's key, 'QUABO_*', and its UID in Redis.
uids_and_rkeys = dict()

def is_acceptable_temperature(temp: float):
    """Returns True only if the provided temperature is between
    MIN_TEMP and MAX_TEMP."""
    return MIN_TEMP <= temp <= MAX_TEMP


def get_adjusted_detector_hv(det_serial_num: str, temp: float) -> float:
    """Given a detector serial number and a temperature in degrees Celsius,
     returns the desired adjusted high-voltage value."""
    try:
        nominal_hv = detector_info[det_serial_num]
    except KeyError as kerr:
        msg = "hv_updater: Failed to get the nominal HV for the detector with serial number: '{0}'."
        msg += "detector_info.json might be missing an entry for this detector. "
        msg += "Error msg: {1}"
        print(msg.format(det_serial_num, kerr))
        raise
    else:
        # Formula from GitHub Issue 47.
        adjusted_voltage = nominal_hv + (temp - 25) * 0.054
        return adjusted_voltage


def update_quabo(quabo_obj: quabo_driver.QUABO,
                 det_serial_nums: list,
                 temp: float):
    """Helper method for the function update_all_quabos. Updates each
     detector in the quabo represented by quabo_obj."""
    for detector_index in range(4):
        try:
            det_serial_num = det_serial_nums[detector_index]
            adjusted_hv = get_adjusted_detector_hv(det_serial_num, temp)
            quabo_obj.hv_set_chan(detector_index, adjusted_hv)
        except KeyError:
            continue


def update_all_quabos(r: redis.Redis):
    """Iterates through each quabo in the observatory and updates
    its detectors' high-voltage values, provided its temperature is
    not too extreme."""
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            module_ip_addr = module['ip_addr']
            for quabo_index in range(4):
                try:
                    uid = module['quabos'][quabo_index]['uid']
                    # Get this quabo's Redis key, if it exists.
                    if uid == '':
                        continue
                    elif uid not in uids_and_rkeys:
                        raise Warning("Quabo %s is not tracked in Redis." % uid)
                    else:
                        quabo_redis_key = uids_and_rkeys[uid]
                    # Get quabo object
                    q_ip_addr = util.quabo_ip_addr(module_ip_addr, quabo_index)
                    quabo_obj = quabo_driver.QUABO(q_ip_addr)
                    # Get the list of detector serial numbers for this quabo.
                    q_info = quabo_info[uid]
                    detector_serial_nums = [s for s in q_info['detector_serialno']]
                    # Get the temperature data for this quabo.
                    temp = float(r.hget(quabo_redis_key, 'TEMP1'))
                except Warning as werr:
                    msg = "hv_updater: Failed to update quabo at index {0} in module {1}."
                    msg += "Error msg: {2} \n"
                    msg += "Attempting to get this quabo's Redis key..."
                    print(msg.format(quabo_index, module_ip_addr, werr))
                    update_inverted_quabo_dict(r)
                    continue
                except AttributeError as aerr:
                    msg = "hv_updater: Failed to update quabo {0} in module {1}. "
                    msg += "Temperature HK data may be missing. "
                    msg += "Error msg: {2}"
                    print(msg.format(quabo_index, module_ip_addr, aerr))
                    continue
                except redis.RedisError as rerr:
                    msg = "hv_updater: A Redis error occurred. "
                    msg += "Error msg: {0}"
                    print(msg.format(rerr))
                    continue
                except KeyError as kerr:
                    msg = "hv_updater: Quabo {0} in module {1} may be missing from a config file."
                    msg += "Error msg: {2}"
                    print(msg.format(quabo_index, module_ip_addr, kerr))
                    continue
                else:
                    # Checks whether the quabo temperature is acceptable.
                    # See https://github.com/panoseti/panoseti/issues/58.
                    if is_acceptable_temperature(temp):
                        update_quabo(quabo_obj, detector_serial_nums, temp)
                    else:
                        msg = "hv_updater: The temperature of quabo {0} in module {1} is {2} C, "
                        msg += "which exceeds the maximum operating temperatures. \n"
                        msg += "Attempting to power down the detectors on this quabo..."
                        print(msg.format(quabo_index, module_ip_addr, temp))
                        try:
                            quabo_obj.hv_set(0)
                            print("Successfully powered down.")
                        except Exception as err:
                            msg = "*** hv_updater: Failed to power down detectors."
                            msg += "Error msg: {0}"
                            print(msg.format(err))
                            continue
                # TODO: Determine when (or if) we should turn detectors back on after a temperature-related power down.


def update_inverted_quabo_dict(r: redis.Redis):
    """Iterates through all quabo keys in Redis and adds [UID]:[Redis key] pairs
    to the dictionary quabo_uids_and_rkeys. Might throw a Redis exception."""
    try:
        for quabo_key in r.keys('QUABO_*'):
            uid = r.hget(quabo_key, 'UID').decode('utf-8')
            # Remove '0x' prefix from Redis UID to match UIDs in config files.
            uid = hex(int(uid, 16))[2:]
            if uid not in uids_and_rkeys:
                uids_and_rkeys[uid] = quabo_key
    except redis.RedisError as err:
        msg = "hv_updater: A Redis error occurred. "
        msg += "Error msg: {0}"
        print(msg.format(err))
        raise


def main():
    """Initializes the script for a delay after being run, waits until Redis
    contains quabo HK data and makes a call to update_all_quabos every
     UPDATE_INTERVAL seconds."""
    r = redis_utils.redis_init()
    print("hv_updater: Waiting for HK to be saved in Redis...")
    time.sleep(UPDATE_INTERVAL)
    update_inverted_quabo_dict(r)
    while len(uids_and_rkeys) == 0:
        print('hv_updater: No quabo data yet. Trying again in %ss...' % UPDATE_INTERVAL)
        time.sleep(UPDATE_INTERVAL)
        update_inverted_quabo_dict(r)
    print("hv_updater: Running...")
    while True:
        update_all_quabos(r)
        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = "hv_updater failed and exited with the error message: '{0}'."
        print(msg.format(e))
        raise
