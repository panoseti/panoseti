#! /usr/bin/env python3
"""
Script that periodically reads each quabo's temperature and
turns off the corresponding module power supply if its temperature
exceeds a specified temperature range.

See https://github.com/panoseti/panoseti/issues/58.
"""

import time
import datetime

import redis

import redis_utils
import config_file
import power
from hv_updater import get_boardloc

# -------------- CONSTANTS --------------- #

# Seconds between updates.
UPDATE_INTERVAL = 30

# Min & max module operating temperatures (degrees Celsius).
MIN_DETECTOR_TEMP = -20.0
MAX_DETECTOR_TEMP = 60.0
MAX_FPGA_TEMP = 85.0

# Get quabo info.
obs_config = config_file.get_obs_config()


def is_acceptable_temperature(temps: (float, float)):
    """Returns a tuple of (TEMP1 is ok?, TEMP2 is ok?) if the corresponding
     sensor temperature is within the specified operating range."""
    temp1_ok = MIN_DETECTOR_TEMP <= temps[0] <= MAX_DETECTOR_TEMP
    temp2_ok = temps[1] <= MAX_FPGA_TEMP
    return temp1_ok, temp2_ok


def get_redis_temps(r: redis.Redis, rkey: str) -> (float, float):
    """
    Given a Quabo's redis key, rkey, returns (TEMP1, TEMP2).
    """
    try:
        temp1 = float(r.hget(rkey, 'TEMP1'))
        temp2 = float(r.hget(rkey, 'TEMP2'))
        return temp1, temp2
    except redis.RedisError as err:
        msg = "module_temp_monitor: A Redis error occurred. "
        msg += "Error msg: {0}"
        print(msg.format(err))
        raise
    except TypeError as terr:
        msg = "module_temp_monitor: Failed to update '{0}'. "
        msg += "Temperature HK data may be missing. "
        msg += "Error msg: {1}"
        print(msg.format(rkey, terr))
        raise


def check_all_module_temps(r: redis.Redis):
    """
    Iterates through each quabo in the observatory, reads its temperature,
    and, if the temperature is too extreme, turns off corresponding module
    power supply.
    """
    for dome in obs_config['domes']:
        for module in dome['modules']:
            module_ip_addr = module['ip_addr']
            # Check whether the UPS socket associated with this module is powered on.
            if power.quabo_power_query():
                for quabo_index in range(4):
                    quabo_obj = None
                    try:
                        # Get this Quabo's redis key.
                        rkey = "QUABO_{0}".format(get_boardloc(module_ip_addr, quabo_index))
                        # Get this Quabo's temp, if it exists.
                        if rkey.encode('utf-8') not in r.keys():
                            raise Warning("%s is not tracked in Redis." % rkey)
                        else:
                            # Get the temperature data for this quabo.
                            temps = get_redis_temps(r, rkey)
                    except Warning as werr:
                        msg = "module_temp_monitor: {0}\n\tFailed to update quabo at index {1} with base IP {2}. "
                        msg += "\tError msg: {3}"
                        print(msg.format(datetime.datetime.now(), quabo_index, module_ip_addr, werr))
                        continue
                    except redis.RedisError as rerr:
                        msg = "module_temp_monitor: {0}\n\tA Redis error occurred. "
                        msg += "\tError msg: {1}"
                        print(msg.format(datetime.datetime.now(), rerr))
                        raise
                    else:
                        # Checks whether the quabo temperatures are acceptable.
                        # See https://github.com/panoseti/panoseti/issues/58.
                        detector_temp_ok, fpga_temp_ok = is_acceptable_temperature(temps)
                        if not detector_temp_ok or not detector_temp_ok:
                            if not detector_temp_ok:
                                msg = "module_temp_monitor: {0}\n\tThe DETECTOR temp of quabo {1} with base IP {2} "
                                msg += " is {3} C, which exceeds the operating temperature range: {4} C to {5} C.\n"
                                msg += "\tAttempting to turn off the power supply for this module..."
                                print(msg.format(datetime.datetime.now(), quabo_index,
                                                 module_ip_addr, temps[0], MIN_DETECTOR_TEMP, MAX_DETECTOR_TEMP))
                            else:
                                msg = "module_temp_monitor: {0}\n\tThe FPGA temp of quabo {1} with base IP {2} "
                                msg += "is {3} C, which exceeds the operating temperature of {4} C.\n"
                                msg += "\tAttempting to turn off the power supply for this module..."
                                print(msg.format(datetime.datetime.now(), quabo_index,
                                                 module_ip_addr, temps[1], MAX_FPGA_TEMP))
                            try:
                                ups = obs_config['ups']
                                url = ups['url']
                                socket = ups['quabo_socket']
                                power.quabo_power(False)
                                msg = "\tSuccessfully turned off power to socket {0} in the UPS with url: {1}."
                                print(msg.format(socket, url))
                                break
                            except Exception as err:
                                msg2 = "*** module_temp_monitor: {0}\n\tFailed to turn off module power supply!"
                                msg += "Error msg: {1}"
                                print(msg.format(datetime.datetime.now(), err))
                                continue


def main():
    """Makes a call to check_all_module_temps every UPDATE_INTERVAL seconds."""
    r = redis_utils.redis_init()
    print("module_temp_monitor: Running...")
    while True:
        check_all_module_temps(r)
        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = "module_temp_monitor: {0} \n\tFailed and exited with the error message: {1}"
        print(datetime.datetime.now(), msg.format(e))
        raise
