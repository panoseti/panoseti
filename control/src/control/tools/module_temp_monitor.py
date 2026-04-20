#!/usr/bin/env python3
"""
module_temp_monitor.py: A daemon that monitors Quabo temperatures.
If temperatures exceed safe operating ranges, it informs the operator and
turns off the corresponding web power switch.
"""

import datetime
import time
from typing import Any

import redis
import redis_utils
from panoseti_grpc.telemetry.logger import get_logger

import control.power as power
from control.utils import config_file
from control.utils.paths import PanoPaths
from control.utils.pydantic_config_models import ObsConfigValidator
from control.utils.util import are_redis_daemons_running

# Safe operating temperature ranges (in Celsius)
MIN_DETECTOR_TEMP = -20
MAX_DETECTOR_TEMP = 60
MAX_FPGA_TEMP = 80

UPDATE_INTERVAL = 10  # Seconds between temperature checks

PanoPaths.logs_dir().mkdir(parents=True, exist_ok=True)
logger = get_logger(service_name='temp_monitor', log_dir=str(PanoPaths.logs_dir()), grpc_enabled=True)


def is_acceptable_temperature(temps: list[float]) -> tuple[bool, bool]:
    """Verify that Quabo temperatures are within safe operating limits.

    Args:
        temps: A list of [detector_temp, fpga_temp] in Celsius.

    Returns:
        A tuple of (detector_temp_ok, fpga_temp_ok).
    """
    detector_temp_ok = MIN_DETECTOR_TEMP <= temps[0] <= MAX_DETECTOR_TEMP
    fpga_temp_ok = temps[1] <= MAX_FPGA_TEMP
    return detector_temp_ok, fpga_temp_ok


def check_all_module_temps(obs_config: ObsConfigValidator | dict[str, Any], wps_to_modules: dict[str, set[str]], r: redis.Redis) -> set[str]:
    """Inspect the temperatures of all active Quabos in the observatory.
    
    Reads current temperatures from Redis and identifies modules that 
    have exceeded safe thresholds.

    Args:
        obs_config: The physical observatory configuration model or dict.
        wps_to_modules: Pre-computed mapping of WPS names to module IP sets.
        r: An active Redis client connection.

    Returns:
        A set of WPS names that should be turned off to protect hardware.
    """
    if isinstance(obs_config, dict):
        obs_config = ObsConfigValidator(**obs_config)

    wps_to_turn_off = set()
    for dome in obs_config.domes:
        for module in dome.modules:
            module_ip_addr = str(module.ip_addr)
            module_wps_key = module.wps or 'wps'
            for quabo_index in range(4):
                try:
                    # Get this Quabo's redis key.
                    rkey = f"QUABO_{config_file.get_boardloc(module_ip_addr, quabo_index)}"
                    # Get this Quabo's temp, if it exists.
                    det_temp = redis_utils.get_casted_redis_value(r, rkey, 'TEMP1')
                    fpga_temp = redis_utils.get_casted_redis_value(r, rkey, 'TEMP2')
                    if det_temp is None or fpga_temp is None:
                        continue
                    temps = [float(det_temp), float(fpga_temp)]
                except (ValueError, TypeError) as werr:
                    msg = "module_temp_monitor: {0}\n\tA parsing error occurred for Quabo {1} at {2}. "
                    msg += "\tError msg: {3}"
                    logger.error(msg.format(datetime.datetime.now(), quabo_index, module_ip_addr, werr))
                    continue
                except redis.RedisError as rerr:
                    msg = "module_temp_monitor: {0}\n\tA Redis error occurred. "
                    msg += "\tError msg: {1}"
                    logger.error(msg.format(datetime.datetime.now(), rerr))
                    raise
                else:
                    # Checks whether the Quabo temperatures are acceptable.
                    detector_temp_ok, fpga_temp_ok = is_acceptable_temperature(temps)
                    # If the detector or fpga temps exceed thresholds, inform the operator and turn off the corresponding wps.
                    if not detector_temp_ok or not fpga_temp_ok:
                        if not detector_temp_ok:
                            msg = "The DETECTOR temp of Quabo {0} is {1} C, which exceeds the operating temperature range: {2} C to {3} C. "
                            logger.info(msg.format(
                                config_file.get_boardloc(module_ip_addr, quabo_index), temps[0], MIN_DETECTOR_TEMP, MAX_DETECTOR_TEMP)
                            )
                            
                        if not fpga_temp_ok:
                            msg = "The FPGA temp of Quabo {0} is {1} C, which exceeds the operating temperature of {2} C. "
                            logger.info(msg.format(
                                config_file.get_boardloc(module_ip_addr, quabo_index), temps[1], MAX_FPGA_TEMP)
                            )
                        logger.info(f'Attempting to turn off the wps: {module_wps_key}')
                        wps_to_turn_off.add(module_wps_key)
    return wps_to_turn_off


def get_wps_to_modules(obs_config: ObsConfigValidator | dict[str, Any]) -> dict[str, set[str]]:
    """Build a mapping of Web Power Switches to their connected modules.

    Args:
        obs_config: The physical observatory configuration model or dict.

    Returns:
        A dictionary mapping WPS unit names (e.g. 'wps', 'wps1') to sets of module IPs.
    """
    if isinstance(obs_config, dict):
        obs_config = ObsConfigValidator(**obs_config)

    wps_to_modules: dict[str, set[str]] = dict()
    for dome in obs_config.domes:
        for module in dome.modules:
            module_ip_addr = str(module.ip_addr)
            module_wps_key = module.wps or 'wps'
            if module_wps_key in wps_to_modules:
                wps_to_modules[module_wps_key].add(module_ip_addr)
            else:
                wps_to_modules[module_wps_key] = {module_ip_addr}
    return wps_to_modules


def update_power(obs_config: ObsConfigValidator | dict[str, Any], wps_to_modules: dict[str, set[str]], wps_to_turn_off: set[str]) -> None:
    """Execute power-off commands for modules that have exceeded safe temperatures.

    Args:
        obs_config: The physical observatory configuration model or dict.
        wps_to_modules: Mapping of WPS names to module IP sets.
        wps_to_turn_off: Set of WPS names identified as needing shutdown.
    """
    if isinstance(obs_config, dict):
        obs_config = ObsConfigValidator(**obs_config)

    for wps_name in wps_to_turn_off:
        try:
            power.do_wps(wps_name, obs_config, 'off')
        except Exception as e:
            logger.error(f"Failed to turn off WPS {wps_name}: {e}")



def main() -> None:
    logger.info('************************************')
    """Makes a call to check_all_module_temps every UPDATE_INTERVAL seconds."""
    obs_config = config_file.get_obs_config()
    wps_to_modules = get_wps_to_modules(obs_config)
    r = redis_utils.redis_init()
    if not are_redis_daemons_running():
        logger.info('Please start redis daemons')
        return
    logger.info("module_temp_monitor: Running...")
    while True:
        time.sleep(UPDATE_INTERVAL)
        wps_to_turn_off = check_all_module_temps(obs_config, wps_to_modules, r)
        update_power(obs_config, wps_to_modules, wps_to_turn_off)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        msg = "module_temp_monitor: {0} \n\tFailed and exited with the error message: {1}"
        logger.error(msg.format(datetime.datetime.now(), e))
        raise
