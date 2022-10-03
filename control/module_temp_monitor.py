#! /usr/bin/env python3
"""
Script that periodically reads each quabo's temperature and
turns off the corresponding module power supply if its temperature
exceeds a safe temperature range.
See https://github.com/panoseti/panoseti/issues/58.

NOTE: this script calls stop.py if any boards or detectors get too hot.
"""

import time, sys
import datetime
import os
import sys
import redis
import redis_utils
import power

from util import are_redis_daemons_running, write_log
from capture_power import get_wps_rkey
sys.path.insert(0, '../util')
import config_file

# Seconds between updates.
UPDATE_INTERVAL = 10

# Min & max module operating temperatures (degrees Celsius).
MIN_DETECTOR_TEMP = -20.0
MAX_DETECTOR_TEMP = 60.0
MAX_FPGA_TEMP = 85.0


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


def log_powered_off_modules(wps_name, wps_to_modules):
    """If power to this module has just been turned off, add its IP to modules_off and inform operator."""
    msg = 'Successfully powered off the wps: {0}. The following quabos are no longer powered: {1}.'
    quabos_off = list()
    for module_ip_addr in wps_to_modules[wps_name]:
        for quabo_index in range(4):
            quabos_off.append(f'QUABO_{config_file.get_boardloc(module_ip_addr, quabo_index)}')
    write_log(msg.format(wps_name, quabos_off))


def update_power(obs_config, wps_to_modules, wps_to_turn_off):
    """Turn off each wps in wps_off, write a log message describing which
    modules and quabos are no longer powered, then stop this script."""
    if wps_to_turn_off:
        # Stop any active runs.
        write_log(f'Running ./stop.py...')
        os.system('./stop.py')
        for wps_name in wps_to_turn_off:
            wps_dict = obs_config[wps_name]
            try:
                power.quabo_power(wps_dict, False)
                log_powered_off_modules(wps_name, wps_to_modules)
            except Exception as err:
                msg = "Failed to turn off the wps: {0}! "
                msg += "Error msg: {1}"
                write_log(msg.format(wps_name, err))
                continue
        sys.exit()


def check_all_module_temps(obs_config, wps_to_modules, r: redis.Redis):
    """
    Iterates through each quabo in the observatory and reads the detector and fpga temperature from Redis.
    If the temperature is too extreme, we turn off the corresponding module power supply.
    """
    # wps_off stores the name of each wps that is powering an over-temperature module.
    wps_to_turn_off = set()
    for dome in obs_config['domes']:
        for module in dome['modules']:
            module_ip_addr = module['ip_addr']
            module_wps_key = module['wps']
            for quabo_index in range(4):
                try:
                    # Get this Quabo's redis key.
                    rkey = f'QUABO_{config_file.get_boardloc(module_ip_addr, quabo_index)}'
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
                    # Checks whether the Quabo temperatures are acceptable.
                    # See https://github.com/panoseti/panoseti/issues/58.
                    detector_temp_ok, fpga_temp_ok = is_acceptable_temperature(temps)
                    # If the detector or fpga temps exceed thresholds, inform the operator and turn off the corresponding wps.
                    if not detector_temp_ok or not fpga_temp_ok:
                        msg = ''
                        if not detector_temp_ok:
                            msg += "The DETECTOR temp of Quabo {0} is {1} C, which exceeds the operating temperature range: {2} C to {3} C. "
                            write_log(msg.format(
                                config_file.get_boardloc(module_ip_addr, quabo_index), temps[0], MIN_DETECTOR_TEMP, MAX_DETECTOR_TEMP)
                            )
                        if not fpga_temp_ok:
                            msg += "The FPGA temp of Quabo {0} is {1} C, which exceeds the operating temperature of {2} C. "
                            write_log(msg.format(
                                config_file.get_boardloc(module_ip_addr, quabo_index), temps[1], MAX_FPGA_TEMP)
                            )
                        msg += f'Attempting to turn off the wps: {module_wps_key}'
                        wps_to_turn_off.add(module_wps_key)
    return wps_to_turn_off


def get_wps_to_modules(obs_config):
    """Dictionary storing pairs of [wps_name]:[set of IPs of the modules connected to this wps]."""
    wps_to_modules = dict()
    for dome in obs_config['domes']:
        for module in dome['modules']:
            module_ip_addr = module['ip_addr']
            module_wps_key = module['wps']
            if module_wps_key in wps_to_modules:
                wps_to_modules[module_wps_key].add(module_ip_addr)
            else:
                wps_to_modules[module_wps_key] = {module_ip_addr}
    return wps_to_modules


def main():
    """Makes a call to check_all_module_temps every UPDATE_INTERVAL seconds."""
    obs_config = config_file.get_obs_config()
    wps_to_modules = get_wps_to_modules(obs_config)
    r = redis_utils.redis_init()
    if not are_redis_daemons_running():
        write_log('Please start redis daemons')
        return
    print("module_temp_monitor: Running...")
    startup = True
    while True:
        time.sleep(UPDATE_INTERVAL)
        wps_to_turn_off = check_all_module_temps(obs_config, wps_to_modules, r)
        update_power(obs_config, wps_to_modules, wps_to_turn_off)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = "module_temp_monitor: {0} \n\tFailed and exited with the error message: {1}"
        write_log(msg.format(datetime.datetime.now(), e))
        raise
