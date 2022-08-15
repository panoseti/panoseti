#! /usr/bin/env python3
"""
Script that periodically reads each quabo's temperature and
turns off the corresponding module power supply if its temperature
exceeds a safe temperature range.
See https://github.com/panoseti/panoseti/issues/58.

NOTE: this script calls stop.py if any boards or detectors get too hot.

This script can be started with:
    ./config.py --temp_monitors
    ./start.py (automatic)

and stopped with:
    ./config.py --stop_temp_monitors
    kill 9 [PID of this script]

"""

import time
import datetime
import os

import redis

import redis_utils
import config_file
import power
from util import get_boardloc, are_redis_daemons_running, write_log
from capture_power import get_ups_rkey

# Seconds between updates.
UPDATE_INTERVAL = 10

# Min & max module operating temperatures (degrees Celsius).
MIN_DETECTOR_TEMP = -20.0
MAX_DETECTOR_TEMP = 60.0
MAX_FPGA_TEMP = 85.0

# Set of modules that have been turned off by this script.
modules_off = set()


def is_acceptable_temperature(temps: (float, float)):
    """
    Returns a tuple of (TEMP1 is ok?, TEMP2 is ok?) if the corresponding
    sensor temperature is within the specified operating range.
    """
    temp1_ok = MIN_DETECTOR_TEMP <= temps[0] <= MAX_DETECTOR_TEMP
    temp2_ok = temps[1] <= MAX_FPGA_TEMP
    return temp1_ok, temp2_ok


def get_redis_temps(r: redis.Redis, rkey: str) -> (float, float):
    """Given a Quabo's redis key, rkey, returns (TEMP1, TEMP2)."""
    try:
        temp1 = redis_utils.get_casted_redis_value(r, rkey, 'TEMP1')
        temp2 = redis_utils.get_casted_redis_value(r, rkey, 'TEMP2')
        return temp1, temp2
    except redis.RedisError as err:
        msg = "module_temp_monitor: A Redis error occurred. "
        msg += "Error msg: {0}"
        write_log(msg.format(err))
        raise
    except TypeError as terr:
        msg = "module_temp_monitor: Failed to update '{0}'. "
        msg += "Temperature HK data may be missing. "
        msg += "Error msg: {1}"
        write_log(msg.format(rkey, terr))
        raise


def check_all_module_temps(obs_config, r: redis.Redis, startup: bool):
    """
    Iterates through each quabo in the observatory and reads the detector and fpga temperature from Redis.
    If the temperature is too extreme, we turn off the corresponding module power supply.
    """
    for dome in obs_config['domes']:
        for module in dome['modules']:
            module_ip_addr = module['ip_addr']
            # Get the UPS status for this module (ON or OFF).
            module_ups_key = module['ups']
            rkey = get_ups_rkey(module_ups_key)
            power_status = 'OFF'
            try:
                power_status = redis_utils.get_casted_redis_value(r, rkey, 'POWER')
            except redis.RedisError as rerr:
                msg = "module_temp_monitor.py: A Redis error occurred. "
                msg += f"Error msg: {rerr}"
                write_log(msg)
                raise
            if power_status == 'OFF':
                # Check if this module has been turned off.
                if startup:
                    modules_off.add(module_ip_addr)
                elif module_ip_addr not in modules_off:
                    # If power to this module has just been turned off, add its IP to modules_off and inform operator.
                    quabos_off = [f'QUABO_{get_boardloc(module_ip_addr, quabo_index)}' for quabo_index in range(4)]
                    msg = 'module_temp_monitor.py: \n\t The module with base IP {0} has been powered off.'
                    msg += '\n\tThe following quabos are no longer powered: {1}'
                    write_log(msg.format(module_ip_addr, quabos_off))
                    modules_off.add(module_ip_addr)
                continue
            elif power_status == 'ON' and module_ip_addr in modules_off:
                # If this module has just been turned on,
                modules_off.remove(module_ip_addr)

            for quabo_index in range(4):
                try:
                    # Get this Quabo's redis key.
                    rkey = f'QUABO_{get_boardloc(module_ip_addr, quabo_index)}'
                    # Get this Quabo's detector and fpga temperatures, if they exist.
                    if rkey.encode('utf-8') not in r.keys():
                        raise Warning("%s is not tracked in Redis." % rkey)
                    else:
                        temps = get_redis_temps(r, rkey)
                except Warning as werr:
                    msg = "module_temp_monitor: {0}\n\tFailed to update quabo at index {1} with base IP {2}. "
                    msg += "\tError msg: {3}"
                    write_log(msg.format(datetime.datetime.now(), quabo_index, module_ip_addr, werr))
                    continue
                except redis.RedisError as rerr:
                    msg = "module_temp_monitor: {0}\n\tA Redis error occurred. "
                    msg += "\tError msg: {1}"
                    write_log(msg.format(datetime.datetime.now(), rerr))
                    raise
                else:
                    # Checks whether the quabo temperatures are acceptable.
                    # See https://github.com/panoseti/panoseti/issues/58.
                    detector_temp_ok, fpga_temp_ok = is_acceptable_temperature(temps)
                    # If detectors exceed thresholds, inform operator and turn off power to corresponding module.
                    if not detector_temp_ok or not fpga_temp_ok:
                        if not detector_temp_ok:
                            msg = "module_temp_monitor: \n\tThe DETECTOR temp of quabo {0} with base IP {1} "
                            msg += " is {2} C, which exceeds the operating temperature range: {3} C to {4} C.\n"
                            msg += "\tAttempting to turn off the power supply for this module..."
                            write_log(msg.format(quabo_index, module_ip_addr, temps[0],
                                                 MIN_DETECTOR_TEMP, MAX_DETECTOR_TEMP))
                        if not fpga_temp_ok:
                            msg = "module_temp_monitor: \n\tThe FPGA temp of quabo {0} with base IP {1} "
                            msg += "is {2} C, which exceeds the operating temperature of {3} C.\n"
                            msg += "\tAttempting to turn off the power supply for this module..."
                            write_log(msg.format(quabo_index, module_ip_addr, temps[1], MAX_FPGA_TEMP))
                        try:
                            # Stop any active runs.
                            write_log(f'\tRunning ./stop.py...')
                            os.system('./stop.py')
                            ups_dict = obs_config[module_ups_key]
                            power.quabo_power(ups_dict, False)
                            break
                        except Exception as err:
                            msg = "*** module_temp_monitor: \n\tFailed to turn off module power supply!"
                            msg += "Error msg: {1}"
                            write_log(msg.format(err))
                            continue


def main():
    """Makes a call to check_all_module_temps every UPDATE_INTERVAL seconds."""
    obs_config = config_file.get_obs_config()
    r = redis_utils.redis_init()
    if not are_redis_daemons_running():
        print('module_temp_monitor.py: please start redis daemons')
        return
    print("module_temp_monitor: Running...")
    startup = True
    while True:
        time.sleep(UPDATE_INTERVAL)
        check_all_module_temps(obs_config, r, startup)
        if startup:
            startup = False


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = "module_temp_monitor: {0} \n\tFailed and exited with the error message: {1}"
        write_log(msg.format(datetime.datetime.now(), e))
        raise
