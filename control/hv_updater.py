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
import util
sys.path.insert(0, '../util')
import config_file

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

# Set of quabos whose detectors have been turned off by this script.
quabos_off = set()

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
    adjusted_hv_values = [0] * 4
    try:
        for detector_index in range(4):
            det_serial_num = det_serial_nums[detector_index]
            adjusted_hv = get_adjusted_detector_hv(det_serial_num, temp)
            # Save int encoding
            adjusted_hv_values[detector_index] = int(adjusted_hv / 0.00114)
    except KeyError as kerr:
        msg = "A detector in the quabo with IP {0} could not be found in the configuration files. "
        msg += "Error message: {1}"
        print(msg.format(quabo_obj.ip_addr, kerr))
        raise
    else:
        quabo_obj.hv_set(adjusted_hv_values)


def get_boardloc(module_ip_addr: str, quabo_index):
    """Given a module ip address and a quabo index, returns the BOARDLOC of
    the corresponding quabo."""
    pieces = module_ip_addr.split('.')
    boardloc = int(pieces[2]) * 256 + int(pieces[3]) + quabo_index
    return boardloc


def get_redis_temp(r: redis.Redis, rkey: str) -> float:
    """Given a Quabo's redis key, rkey, returns the field value of TEMP1 in Redis."""
    try:
        temp = float(r.hget(rkey, 'TEMP1'))
        return temp
    except redis.RedisError as err:
        msg = "hv_updater: A Redis error occurred. "
        msg += "Error msg: {0}"
        print(msg.format(err))
        raise
    except TypeError as terr:
        msg = "hv_updater: Failed to update '{0}'. "
        msg += "Temperature HK data may be missing. "
        msg += "Error msg: {1}"
        print(msg.format(rkey, terr))
        raise


def update_all_quabos(r: redis.Redis):
    """Iterates through each quabo in the observatory and updates
    its detectors' high-voltage values, provided its temperature is
    not too extreme."""
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            module_ip_addr = module['ip_addr']
            for quabo_index in range(4):
                quabo_obj = None
                try:
                    # Get this Quabo's redis key.
                    rkey = "QUABO_{0}".format(get_boardloc(module_ip_addr, quabo_index))
                    if rkey in quabos_off:
                        continue
                    uid = module['quabos'][quabo_index]['uid']
                    if uid == '':
                        continue
                    # Get this Quabo's temp, if it exists.
                    if rkey.encode('utf-8') not in r.keys():
                        raise Warning("%s is not tracked in Redis." % rkey)
                    else:
                        # Get the temperature data for this quabo.
                        temp = get_redis_temp(r, rkey)
                    # Get quabo object
                    q_ip_addr = config_file.quabo_ip_addr(module_ip_addr, quabo_index)
                    quabo_obj = quabo_driver.QUABO(q_ip_addr)
                    # Get the list of detector serial numbers for this quabo.
                    q_info = quabo_info[uid]
                    detector_serial_nums = [s for s in q_info['detector_serialno']]
                except Warning as werr:
                    msg = "hv_updater: Failed to update quabo at index {0} with base IP {1}. "
                    msg += "Error msg: {2} \n"
                    print(msg.format(quabo_index, module_ip_addr, werr))
                    continue
                except redis.RedisError as rerr:
                    msg = "hv_updater: A Redis error occurred. "
                    msg += "Error msg: {0}"
                    print(msg.format(rerr))
                    raise
                except KeyError as kerr:
                    msg = "hv_updater: Quabo {0} with base IP {1} may be missing from a config file. "
                    msg += "Error msg: {2}"
                    print(msg.format(quabo_index, module_ip_addr, kerr))
                    raise
                else:
                    # Checks whether the quabo temperature is acceptable.
                    # See https://github.com/panoseti/panoseti/issues/58.
                    if is_acceptable_temperature(temp):
                        update_quabo(quabo_obj, detector_serial_nums, temp)
                    else:
                        msg = "hv_updater: The temperature of quabo {0} with base IP {1} is {2} C, "
                        msg += "which exceeds the maximum operating temperatures. \n"
                        msg += "Attempting to power down the detectors on this quabo..."
                        print(msg.format(quabo_index, module_ip_addr, temp))
                        try:
                            quabo_obj.hv_set([0] * 4)
                            quabos_off.add(rkey)
                            print("Successfully powered down.")
                        except Exception as err:
                            msg = "*** hv_updater: Failed to power down detectors. "
                            msg += "Error msg: {0}"
                            print(msg.format(err))
                            continue
                # TODO: Determine when (or if) we should turn detectors back on after a temperature-related power down.
                finally:
                    if quabo_obj is not None:
                        quabo_obj.close()


def main():
    """Makes a call to update_all_quabos every UPDATE_INTERVAL seconds."""
    r = redis_utils.redis_init()
    print("hv_updater: Running...")
    while True:
        update_all_quabos(r)
        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = "hv_updater failed and exited with the error message: {0}"
        print(msg.format(e))
        raise
