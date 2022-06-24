#! /usr/bin/env python3
"""
Script that periodically reads all quabo temperatures and
turns off the corresponding module power supply if the temperature
exceeds a specified threshold.

See https://github.com/panoseti/panoseti/issues/58.
"""

import time
import datetime

import redis

import redis_utils
import config_file
import power
from hv_updater import get_boardloc, get_redis_temp

# -------------- CONSTANTS --------------- #

# Seconds between updates.
UPDATE_INTERVAL = 30

# Min & max module operating temperatures (degrees Celsius).
MIN_TEMP = -20.0
MAX_TEMP = 60.0

# Get quabo info.
obs_config = config_file.get_obs_config()


def is_acceptable_temperature(temp: float):
    """Returns True only if the provided temperature is between
    MIN_TEMP and MAX_TEMP."""
    return MIN_TEMP <= temp <= MAX_TEMP


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
                            temp = get_redis_temp(r, rkey)
                    except Warning as werr:
                        msg = "hv_updater: {0} \n\tFailed to update quabo at index {1} with base IP {2}. "
                        msg += "\tError msg: {3} \n"
                        print(msg.format(datetime.datetime.now(), quabo_index, module_ip_addr, werr))
                        continue
                    except redis.RedisError as rerr:
                        msg = "hv_updater: {0}\n\tA Redis error occurred. "
                        msg += "\tError msg: {1}"
                        print(msg.format(datetime.datetime.now(), rerr))
                        raise
                    else:
                        # Checks whether the quabo temperature is acceptable.
                        # See https://github.com/panoseti/panoseti/issues/58.
                        if not is_acceptable_temperature(temp):
                            msg = "module_temp_monitor: {0} \n\tThe temperature of quabo {1} with base IP {2} is {3} C, "
                            msg += "which exceeds the operating temperature range: {4} C to {5} C. \n"
                            msg += "\tAttempting to turn off the power supply for this module..."
                            print(msg.format(datetime.datetime.now(), quabo_index,
                                             module_ip_addr, temp, MIN_TEMP, MAX_TEMP))
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
                    # TODO: Determine when (or if) we should turn detectors back on after a temperature-related power down.


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
