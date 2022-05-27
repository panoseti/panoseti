#! /usr/bin/env python3

###############################################################################
# Script for periodically updating the high-voltage values in every detector
# active in the observatory. The adjustments are based on real-time temperature
# data collected by housekeeping programs and detector settings specified by
# the detector manufacturer, Hamamatsu.
# See https://github.com/panoseti/panoseti/issues/47.
###############################################################################
import time

import redis

import redis_utils
import quabo_driver
import config_file
import util

#-------------- CONSTANTS ---------------#

# Seconds between updates.
UPDATE_INTERVAL = 5

# Acceptable voltage vals. Currently, all voltages are considered acceptable.
# TODO: replace with real values:
MIN_HV = float('-inf')
MAX_HV = float('inf')

# Acceptable temperature ranges. Currently, all temperatures are considered acceptable.
# TODO: replace with real values:
MIN_TEMP = float('-inf')
MAX_TEMP = float('inf')

#--------- Implementation Globals --------#

# Get quabo and detector info.
quabo_info = config_file.get_quabo_info()
detector_info = config_file.get_detector_info()
quabo_uids = config_file.get_quabo_uids()

# Dict inverting the relation between a quabo's key, 'QUABO_*', and its UID in Redis.
uids_and_rkeys = dict()


def is_acceptable_voltage(adjusted_voltage: float):
    """Returns True only if the proposed voltage adjustment is between
     MIN_HV and MAX_HV."""
    return MIN_HV <= adjusted_voltage <= MIN_HV


def is_acceptable_temperature(temp: float):
    """Returns True only if the provided temperature is between
    MIN_TEMP and MAX_TEMP."""
    return MIN_TEMP <= temp <= MAX_TEMP



def get_adjusted_detector_hv(det_serial_num: str, temp: float) -> float:
    """Given a detector serial number and a temperature in degrees Celsius,
     returns the desired adjusted high-voltage value only if it is within
     an acceptable range."""
    nominal_hv = detector_info[det_serial_num]
    # Formula from GitHub Issue 47.
    adjusted_voltage = nominal_hv + (temp - 25) * 0.054
    if not is_acceptable_voltage(adjusted_voltage):
        msg = "hv_updater: The proposed voltage of {0} for detector {1} is out "
        msg += "of the acceptable range and will not be set. "
        raise Warning(msg.format(adjusted_voltage, det_serial_num))
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
        except Warning as werr:
            print(werr)
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
                    # Get corresponding Redis key if it is tracked in Redis.
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
                    # Note: currently the key 'TEMP1' does not exist in the HK Redis,
                    # so line 113 might throw some kind of error.
                    # Get the temperature data for this quabo.
                    temp = r.hget(quabo_redis_key, 'TEMP1')  # TODO: save temperature HK data in capture_hk.py.
                    temp = float(temp.decode('utf-8'))
                except Warning as werr:
                    msg = "hv_updater: Failed to update quabo {0} "
                    msg += "in module {1}. Error msg: '{2}'"
                    print(msg.format(quabo_index, module_ip_addr, werr))
                    continue
                except AttributeError as aerr:
                    msg = "hv_updater: Failed to update quabo {0} "
                    msg += "in module {1}. Temperature HK data may be "
                    msg += "missing. Error msg: {2}."
                    print(msg.format(quabo_index, module_ip_addr, aerr))
                    raise
                except redis.RedisError as rerr:
                    print("hv_updater: A Redis error occurred."
                          + " Error msg: '{0}'".format(rerr))
                    raise
                except KeyError as kerr:
                    msg = "hv_updater: Quabo {0} in module {1} "
                    msg += "may be missing from a config file. Error msg: {2}"
                    print(msg.format(quabo_index, module_ip_addr, kerr))
                    raise
                else:
                    # Checks whether the Quabo temperature is acceptable.
                    # See https://github.com/panoseti/panoseti/issues/58.
                    if is_acceptable_temperature(temp):
                        # Call helper to adjust detector voltages in this quabo.
                        update_quabo(quabo_obj, detector_serial_nums, temp)
                    else:
                        # TODO: define behavior when the temp is extreme.
                        ...


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
        print("hv_updater: A Redis error occurred."
              + " Error msg: '{0}'".format(err))
        raise


def main():
    """Initializes the script for a delay after being run, waits until Redis
    contains quabo HK data and makes a call to update_all_quabos every
     UPDATE_INTVERAL seconds."""
    r = redis_utils.redis_init()
    print("hv_updater: Waiting for HK to be saved in Redis...")
    time.sleep(UPDATE_INTERVAL) # Not sure how long it takes for Hk data to start being collected.
    update_inverted_quabo_dict(r)
    while len(uids_and_rkeys) == 0:
        print('hv_updater: No quabo data yet.'
              + ' Trying again in %ss...' % UPDATE_INTERVAL)
        update_inverted_quabo_dict(r)
        time.sleep(UPDATE_INTERVAL)
    print("hv_updater: Running...")
    while True:
        update_all_quabos(r)
        time.sleep(UPDATE_INTERVAL)


try:
    main()
except Exception as e:
    print(e)
    if __name__ == "__main__":
        raise
    else:
        pass
