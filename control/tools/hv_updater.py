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
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import time
from typing import Any

import redis

from driver import quabo_driver
from utils import config_file, redis_utils, util

#-------------- CONSTANTS ---------------#
# HV offset
HV_OFFSET = 1.073

# Seconds between updates.
UPDATE_INTERVAL = 3

# Min & max detector operating temperatures (degrees Celsius).
MIN_TEMP = -20.0
MAX_TEMP = 60.0
# Min & max hv.
MIN_HV = 0
MAX_HV = 60

#--------- Implementation Globals --------#

# Get quabo and detector info.
quabo_info = config_file.get_quabo_info()
detector_info = config_file.get_detector_info()
quabo_uids = config_file.get_quabo_uids()
data_config = config_file.get_data_config()
network_config = config_file.get_network_config()
# Set of quabos whose detectors have been turned off by this script.
quabos_off = set()

def is_acceptable_temperature(temp: float) -> bool:
    """Returns True only if the provided temperature is between
    MIN_TEMP and MAX_TEMP."""
    return MIN_TEMP <= temp <= MAX_TEMP

def is_acceptable_hv(monitored_hv: list[float]) -> bool:
    """Returns True only if the provided hv is reasonable. """
    r = True
    for hv in monitored_hv:
        r = MIN_HV <= hv <= MAX_HV
    return r

def get_adjusted_detector_hv(det_serial_num: str, temp: float) -> float:
    """Given a detector serial number and a temperature in degrees Celsius,
     returns the desired adjusted high-voltage value."""
    logger = logging.getLogger('PANOSETI.HVUpdater')
    try:
        nominal_hv = detector_info[str(det_serial_num)]
    except KeyError as kerr:
        msg = "hv_updater: Failed to get the nominal HV for the detector with serial number: '{0}'."
        msg += "detector_info.json might be missing an entry for this detector. "
        msg += "Error msg: {1}"
        logger.error(msg.format(det_serial_num, kerr))
        raise
    else:
        # Formula from GitHub Issue 47.
        adjusted_voltage = nominal_hv + (temp - 25) * 0.054
        return adjusted_voltage


def update_quabo(quabo_obj: quabo_driver.QUABO,
                 rkey: str,
                 quabo_status: dict[str, Any]
                ) -> None:
    """Helper method for the function update_all_quabos. Updates each
     detector in the quabo represented by quabo_obj."""
    logger = logging.getLogger('PANOSETI.HVUpdater')
    adjusted_hv_values = [0] * 4
    # get the metadata from the quabo_status dict
    adjusted_hv = quabo_status[rkey]['adjusted_hv']
    temp = quabo_status[rkey]['temp']
    det_serial_nums = quabo_status[rkey]['detector_serial_nums']
    monitored_hv = quabo_status[rkey]['monitored_hv']
    monitored_det_cur = quabo_status[rkey]['monitored_det_cur']
    init_set = quabo_status[rkey]['init_set']
    try:
        logger.debug(f'           Temp: {temp:.02f}')
        for detector_index in range(4):
            det_serial_num = det_serial_nums[detector_index]
            # compensate for current limitter(LND150) voltage drop
            # the typical resistance of LND150 is 850 ohm
            target_hv = get_adjusted_detector_hv(det_serial_num, temp) + monitored_det_cur[detector_index] * 850/4
            if init_set:
                # When we set HV the first time, we don't use the loop
                adjusted_hv[detector_index] = target_hv
                logger.debug(f'{rkey} Init HV setting.')
            else:
                # Now, we start to use the loop the set the HV
                adjusted_hv[detector_index] = adjusted_hv[detector_index] + (target_hv - monitored_hv[detector_index])*0.5
                # make sure the adjusted_hv is a reasonable value
                if adjusted_hv[detector_index] >= MAX_HV:
                    adjusted_hv[detector_index] = MAX_HV
                #adjusted_hv = target_hv
            logger.debug(f'     Target HV{detector_index}: -{target_hv:.3f}V')
            logger.debug(f'  Monitored HV{detector_index}: -{monitored_hv[detector_index]:.3f}V')
            logger.debug(f'  Detector Cur{detector_index}:  {monitored_det_cur[detector_index]*1000:.3f}mA')
            logger.debug(f'   Adjusted HV{detector_index}: -{adjusted_hv[detector_index]:.3f}V')
            logger.debug(' ')
            # Save int encoding
            adjusted_hv_values[detector_index] = int((adjusted_hv[detector_index] + HV_OFFSET) / 0.0011453)
            #adjusted_hv_values[detector_index] = int((adjusted_hv[detector_index]) / 0.0011324717)
    except KeyError as kerr:
        msg = "A detector in the quabo with IP {0} could not be found in the configuration files. "
        msg += "Error message: {1}"
        logger.error(msg.format(quabo_obj.ip_addr, kerr))
        raise
    else:
        quabo_obj.hv_set(adjusted_hv_values)
        if init_set:
            time.sleep(5)
            quabo_status[rkey]['init_set'] = False


def get_redis_temp(r: redis.Redis, rkey: str) -> float:
    """Given a Quabo's redis key, rkey, returns the field value of TEMP1 in Redis."""
    logger = logging.getLogger('PANOSETI.HVUpdater')
    try:
        val = r.hget(rkey, 'TEMP1')
        if val is None:
            raise TypeError("TEMP1 is None")
        temp = float(val.decode('utf-8') if isinstance(val, bytes) else str(val))
        return temp
    except redis.RedisError as err:
        msg = "hv_updater: A Redis error occurred. "
        msg += "Error msg: {0}"
        logger.error(msg.format(err))
        raise
    except TypeError as terr:
        msg = "hv_updater: Failed to update '{0}'. "
        msg += "Temperature HK data may be missing. "
        msg += "Error msg: {1}"
        logger.error(msg.format(rkey, terr))
        raise

def get_redis_hv(r: redis.Redis, rkey: str, q: int) -> float:
    """Given a Quabo's redis key, rkey, returns the field value of HVMON{q} in Redis."""
    logger = logging.getLogger('PANOSETI.HVUpdater')
    try:
        val = r.hget(rkey, f'HVMON{q}')
        if val is None:
            raise TypeError(f"HVMON{q} is None")
        hv = float(val.decode('utf-8') if isinstance(val, bytes) else str(val)) * -1
        return hv
    except redis.RedisError as err:
        msg = "hv_updater: A Redis error occurred. "
        msg += "Error msg: {0}"
        logger.error(msg.format(err))
        raise
    except TypeError as terr:
        msg = "hv_updater: Failed to update '{0}'. "
        msg += f"HV{q} HK data may be missing. "
        msg += "Error msg: {1}"
        logger.error(msg.format(rkey, terr))
        raise

def get_redis_det_current(r: redis.Redis, rkey: str, q: int) -> float:
    """Given a Quabo's redis key, rkey, returns the field value of DETR{q}_CURR in Redis."""
    logger = logging.getLogger('PANOSETI.HVUpdater')
    try:
        val = r.hget(rkey, f'DETR{q}_CURR')
        if val is None:
            raise TypeError(f"DETR{q}_CURR is None")
        det_cur = float(val.decode('utf-8') if isinstance(val, bytes) else str(val))
        return det_cur
    except redis.RedisError as err:
        msg = "hv_updater: A Redis error occurred. "
        msg += "Error msg: {0}"
        logger.error(msg.format(err))
        raise
    except TypeError as terr:
        msg = "hv_updater: Failed to update '{0}'. "
        msg += f"HV{q} HK data may be missing. "
        msg += "Error msg: {1}"
        logger.error(msg.format(rkey, terr))
        raise

def check_timestamp(r: redis.Redis, rkey: str, quabo_status: dict[str, Any]) -> bool:
    """
        check the timestamp in the redis database.
        if the timestamp doesn't change, return false
    """
    logger = logging.getLogger('PANOSETI.HVUpdater')
    try:
        val = r.hget(rkey, 'Computer_UTC')
        if val is None:
            return False
        timestamp = float(val.decode('utf-8') if isinstance(val, bytes) else str(val))
        if timestamp - quabo_status[rkey]['timestamp_pre'] > 0.1 :
            # check the timestamp difference, to make sure the hk data is updated
            quabo_status[rkey]['timestamp_pre'] = timestamp
            return True
        else:
            return False
    except redis.RedisError as err:
        msg = "hv_updater: A Redis error occurred. "
        msg += "Error msg: {0}"
        logger.error(msg.format(err))
        raise

def init_quabo_status(rkey: str, quabo_status: dict[str, Any]) -> None:
    """init the quabo info for the specific Quabo. """
    quabo_status[rkey] = {
        'init_set' : True,
        'timestamp_pre' : 0,
        'detector_serial_nums' : [0, 0, 0, 0],
        'temp' : 0,
        'monitored_hv' : [0, 0, 0, 0],
        'monitored_det_cur' : [0, 0, 0, 0],
        'adjusted_hv' : [0, 0, 0, 0],
    }

def update_all_quabos(r: redis.Redis, quabo_status: dict[str, Any]) -> None:
    """Iterates through each quabo in the observatory and updates
    its detectors' high-voltage values, provided its temperature is
    not too extreme."""
    logger = logging.getLogger('PANOSETI.HVUpdater')
    for dome in quabo_uids.domes:
        for module in dome.modules:
            module_ip_addr = str(module.ip_addr)
            for quabo_index in range(4):
                quabo_obj = None
                try:
                    # Get this Quabo's redis key.
                    rkey = f"QUABO_{config_file.get_boardloc(module_ip_addr, quabo_index)}"
                    # check if we already have records for this Quabo
                    if rkey not in quabo_status:
                        # if not, we need to create a new record for this Quabo
                        init_quabo_status(rkey, quabo_status)
                    if rkey in quabos_off:
                        continue
                    uid = module.quabos[quabo_index].uid
                    if uid == '':
                        continue
                    if not check_timestamp(r, rkey, quabo_status):
                        logger.warning("Housekeeping data hasn't been updated.")
                        continue
                    # Get this Quabo's temp, if it exists.
                    all_keys = r.keys()
                    if not isinstance(all_keys, list) or rkey.encode('utf-8') not in all_keys:
                        raise Warning(f"{rkey} is not tracked in Redis.")
                    else:
                        # Get the temperature data for this quabo.
                        temp = get_redis_temp(r, rkey)
                        quabo_status[rkey]['temp'] = temp
                        # Get the monitored HV
                        monitored_hv = []
                        for i in range(4):
                            mhv = get_redis_hv(r, rkey, i)
                            monitored_hv.append(mhv)
                        quabo_status[rkey]['monitored_hv'] = monitored_hv
                        # Get the monitored detector current
                        monitored_det_cur = []
                        for i in range(4):
                            mdetcur = get_redis_det_current(r, rkey, i)
                            monitored_det_cur.append(mdetcur)
                        quabo_status[rkey]['monitored_det_cur'] = monitored_det_cur
                    # Get quabo object
                    q_ip_addr = config_file.quabo_ip_addr(module_ip_addr, quabo_index)
                    ip_port = util.get_quabo_ip_port(module_ip_addr, quabo_index, network_config)
                    real_ip = ip_port['ip_addr']
                    port = ip_port['cmd_port']
                    logger.debug('-------------------------')
                    logger.debug(f'{rkey}({q_ip_addr}):')
                    logger.debug(f'Port forwarding: {real_ip}:{port}')
                    quabo_obj = quabo_driver.QUABO(real_ip, port)
                    # Get the list of detector serial numbers for this quabo.
                    try:
                        q_info = quabo_info[uid]
                    except Exception:
                        q_info = quabo_info['default']
                        logger.warning(f'No calibration data: UID - {uid}')
                    detector_serial_nums = [s for s in q_info['detector_serialno']]
                    # record the detector_serial_nums in the quabo_status dict
                    quabo_status[rkey]['detector_serial_nums'] = detector_serial_nums
                except Warning as werr:
                    msg = "hv_updater: Failed to update quabo at index {0} with base IP {1}. "
                    msg += "Error msg: {2} \n"
                    logger.error(msg.format(quabo_index, module_ip_addr, werr))
                    continue
                except redis.RedisError as rerr:
                    msg = "hv_updater: A Redis error occurred. "
                    msg += "Error msg: {0}"
                    logger.error(msg.format(rerr))
                    raise
                except KeyError as kerr:
                    msg = "hv_updater: Quabo {0} with base IP {1} may be missing from a config file. "
                    msg += "Error msg: {2}"
                    logger.error(msg.format(quabo_index, module_ip_addr, kerr))
                    raise
                except OSError:
                    continue
                else:
                    # Checks whether the quabo temperature is acceptable.
                    # See https://github.com/panoseti/panoseti/issues/58.
                    #if is_acceptable_temperature(temp) and is_acceptable_hv(monitored_hv):
                    if is_acceptable_temperature(temp):
                        update_quabo(quabo_obj, rkey, quabo_status)
                    else:
                        msg = "hv_updater: The temperature of quabo {0} with base IP {1} is {2} C, "
                        msg += "which exceeds the maximum operating temperatures. \n"
                        msg += "Attempting to power down the detectors on this quabo..."
                        logger.info(msg.format(quabo_index, module_ip_addr, temp))
                        try:
                            quabo_obj.hv_set([0] * 4)
                            quabos_off.add(rkey)
                            logger.info('Successfully powered down.')
                        except Exception as err:
                            msg = "*** hv_updater: Failed to power down detectors. "
                            msg += "Error msg: {0}"
                            logger.error(msg.format(err))
                            continue
                # TODO: Determine when (or if) we should turn detectors back on after a temperature-related power down.
                finally:
                    if quabo_obj is not None:
                        quabo_obj.close()

def main() -> None:
    """Makes a call to update_all_quabos every UPDATE_INTERVAL seconds."""
    r = redis_utils.redis_init()
    print("hv_updater: Running...")
    """
    The dict is used for recording all of the hv status data for each quabo
    It contains the following info:
    1. temp - temperature;
    2. timestamp_pre - the previous timestamp for the Quabo;
    3. init_set - it shows if the init setting is done;
    4. monitored_hv - this is a list, which records the monitored 4 hv on the quabo;
    5. monitored_hvi - this is a list, which records the monitored 4 hv  current on the quabo;
    5. adjusted_hv - this is a list, which records the latest 4 hv on the quabo.
    """
    quabo_status: dict[str, Any] = {}
    while True:
        update_all_quabos(r, quabo_status)
        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logfile = 'logs/hv_updater.log'
    util.create_logger(logfile, 'PANOSETI.HVUpdater', 'a')
    logger = logging.getLogger('PANOSETI.HVUpdater')
    logger.info('************************************')
    if data_config.detector_overvoltage is None:
        logger.warning('detector_overvoltage is not set in data_config.json')
        logger.warning('Use the default overvoltage: 3V.')
    else:
        logger.info(f"Use the overvoltage: {data_config.detector_overvoltage}")
    try:
        main()
    except Exception as e:
        msg = "hv_updater failed and exited with the error message: {0}"
        logger.error(msg.format(e))
        raise
