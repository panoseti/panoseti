from pathlib import Path
import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from rich import print
from rich.pretty import pprint
import sys
import json
import re

# Import utils from other panoseti directories
repo_root = Path('..')
util_path = repo_root / 'util'
control_utils = repo_root / 'control/utils'
for p in [util_path, control_utils]:
    sys.path.append(str(p))

import pff, config_file, pixel_coords, util



def create_quabo_install_df(obs_config: dict, quabo_uids: dict, detovervoltage: float) -> pd.DataFrame:
    """
    Creates the quabo_install_df from configuration files.

    Args:
        obs_config (dict): A dictionary representing the obs_config.json file.
        quabo_uids (dict): A dictionary representing the quabo_uids.json file.
        detovervoltage (float): The detector overvoltage value for the observing run.

    Returns:
        pd.DataFrame: A DataFrame with the quabo installation information.
    """
    records = []

    # Create a mapping from IP address to quabo UIDs
    ip_to_uids = {}
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            ip_to_uids[module['ip_addr']] = module['quabos']

    # Iterate through the obs_config to build records
    for dome in obs_config['domes']:
        dome_name = dome['name']
        for module in dome['modules']:
            module_ip_addr = module['ip_addr']
            mobo_serial_no = module['mobo_serialno']

            if module_ip_addr in ip_to_uids:
                for i, quabo in enumerate(ip_to_uids[module_ip_addr]):
                    record = {
                        'dome': dome_name,
                        'module_ip_addr': module_ip_addr,
                        'mobo_serial_no': mobo_serial_no,
                        'quabo_uid': quabo['uid'],
                        'quabo_num': i,
                        'detector_overvoltage': detovervoltage
                    }
                    records.append(record)

    df = pd.DataFrame(records)
    # Set the primary key as specified in the schema
    # df.set_index('quabo_uid', inplace=True)
    return df


def create_detector_install_df(quabo_info: dict) -> pd.DataFrame:
    """
    Creates the detector_install_df from the quabo_info file.

    Args:
        quabo_info (dict): A dictionary representing the quabo_info.json file.

    Returns:
        pd.DataFrame: A DataFrame with the detector installation information.
    """
    records = []
    for board in quabo_info:
        # Skip default entry if it exists
        if board.get('uid') == 'default':
            continue

        quabo_uid = board['uid']
        board_version = board['board_version']
        quabo_serialno_str = board['serialno']

        # Extract serial number integer using the specified regex
        match = re.search(r"SN0*(\d+)", quabo_serialno_str)
        quabo_serialno = int(match.group(1)) if match else None

        for i, detector_sn in enumerate(board['detector_serialno']):
            record = {
                'quabo_uid': quabo_uid,
                'board_version': board_version,
                'quabo_serialno_str': quabo_serialno_str,
                'quabo_serialno': quabo_serialno,
                'detector_serialno': detector_sn,
                'detector_quadrant': i
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df


def create_pixel_ph_baseline_df(quabo_ph_baseline: dict) -> pd.DataFrame:
    """
    Creates the pixel_ph_baseline_df from the pulse-height baseline file.

    Args:
        quabo_ph_baseline (dict): A dictionary representing the quabo_ph_baseline.json file.

    Returns:
        pd.DataFrame: A DataFrame with pixel pulse-height baseline information.
    """
    records = []
    for quabo_data in quabo_ph_baseline['quabos']:
        quabo_uid = quabo_data['uid']
        for i, coef in enumerate(quabo_data['coefs']):
            # As per documentation, the 256 pixels are ordered by quadrant.
            # First 64 are quadrant 0, next 64 are quadrant 1, etc.
            detector_quadrant = i // 64
            record = {
                'quabo_uid': quabo_uid,
                'coefs_idx': i,
                'detector_quadrant': detector_quadrant,
                'baseline_adc': coef
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df


def create_detector_ph_calibration_df(quabo_calib: dict, quabo_info: dict, board_serialno_str: str,
                                      detovervoltage: float) -> pd.DataFrame:
    """
    Creates the detector_ph_calibration_df from a specific calibration file.

    Args:
        quabo_calib (dict): Dict from a quabo_calib_[serialno].json file.
        quabo_info (dict): Dict from the main quabo_info.json file.
        board_serialno_str (str): The serial number string for the board this calib file applies to (e.g., "SN037").
        detovervoltage (float): The detector overvoltage value.

    Returns:
        pd.DataFrame: A DataFrame with the detector pulse-height calibration coefficients.
    """
    # Find the quabo_uid corresponding to the board serial number
    quabo_uid = None
    for board in quabo_info:
        if board.get('serialno') == board_serialno_str:
            quabo_uid = board['uid']
            break

    if quabo_uid is None:
        raise ValueError(f"Could not find quabo_uid for serial number {board_serialno_str}")

    records = []
    for i, quadrant_data in enumerate(quabo_calib['quadrants']):
        record = {
            'quabo_uid': quabo_uid,
            'detovervol': detovervoltage,
            'detector_serialno': quadrant_data['detserial'],
            'detector_quadrant': i,
            'm': quadrant_data['m'],
            'n': quadrant_data['n'],
            'a': quadrant_data['a'],
            'b': quadrant_data['b'],
            'ah': quadrant_data['ah'],
            'bh': quadrant_data['bh']
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df


def create_pixel_ph_gain_delta_df(quabo_calib: dict, quabo_info: dict, board_serialno_str: str,
                                  detovervoltage: float) -> pd.DataFrame:
    """
    Creates the pixel_ph_gain_delta_df from a specific calibration file.

    Args:
        quabo_calib (dict): Dict from a quabo_calib_[serialno].json file.
        quabo_info (dict): Dict from the main quabo_info.json file.
        board_serialno_str (str): The serial number string for the board this calib file applies to (e.g., "SN037").
        detovervoltage (float): The detector overvoltage value.

    Returns:
        pd.DataFrame: A DataFrame with the pixel pulse-height gain delta calibrations.
    """
    # Find the quabo_uid corresponding to the board serial number
    quabo_uid = None
    for board in quabo_info:
        if board.get('serialno') == board_serialno_str:
            quabo_uid = board['uid']
            break

    if quabo_uid is None:
        raise ValueError(f"Could not find quabo_uid for serial number {board_serialno_str}")

    records = []
    pixel_gain_matrix = quabo_calib['pixel_gain']
    for key, row in enumerate(pixel_gain_matrix):
        for idx, gain_delta in enumerate(row):
            record = {
                'quabo_uid': quabo_uid,
                'detovervol': detovervoltage,
                'pixel_gain_key': key,
                'pixel_gain_idx': idx,
                'gain_delta': gain_delta
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df


# --- Example Usage ---
if __name__ == '__main__':
    # Assume the attached JSON files have been loaded into dictionaries.
    # I will load them here from the provided attachments for demonstration.
    # NOTE: You mentioned you would handle the path logic.
    with open('obs_config.json', 'r') as f:
        obs_config = json.load(f)
    with open('quabo_uids.json', 'r') as f:
        quabo_uids = json.load(f)
    with open('quabo_info-checkpoint.json', 'r') as f:
        quabo_info = json.load(f)  # Using the checkpoint version as it has more entries
    with open('quabo_ph_baseline.json', 'r') as f:
        quabo_ph_baseline = json.load(f)
    with open('quabo_calib_37.json', 'r') as f:
        quabo_calib_37 = json.load(f)

    # The detector overvoltage is specified in obs_config.json
    DETOVERVOLTAGE = obs_config.get('detector_overvoltage', 2.0)

    print("--- Creating quabo_install_df ---")
    quabo_install_df = create_quabo_install_df(obs_config, quabo_uids, DETOVERVOLTAGE)
    print(quabo_install_df.head())
    print("\n")

    print("--- Creating detector_install_df ---")
    detector_install_df = create_detector_install_df(quabo_info)
    print(detector_install_df.head())
    print("\n")

    print("--- Creating pixel_ph_baseline_df ---")
    pixel_ph_baseline_df = create_pixel_ph_baseline_df(quabo_ph_baseline)
    print(pixel_ph_baseline_df.head())
    print("\n")

    # For the calibration DFs, we process one calib file at a time.
    # The file quabo_calib_37.json corresponds to board "SN037"
    BOARD_SERIALNO_STR = "SN037"
    print(f"--- Creating calibration DFs for board {BOARD_SERIALNO_STR} ---")

    print("--- Creating detector_ph_calibration_df ---")
    detector_ph_calibration_df = create_detector_ph_calibration_df(
        quabo_calib=quabo_calib_37,
        quabo_info=quabo_info,
        board_serialno_str=BOARD_SERIALNO_STR,
        detovervoltage=DETOVERVOLTAGE
    )
    print(detector_ph_calibration_df.head())
    print("\n")

    print("--- Creating pixel_ph_gain_delta_df ---")
    pixel_ph_gain_delta_df = create_pixel_ph_gain_delta_df(
        quabo_calib=quabo_calib_37,
        quabo_info=quabo_info,
        board_serialno_str=BOARD_SERIALNO_STR,
        detovervoltage=DETOVERVOLTAGE
    )
    print(pixel_ph_gain_delta_df.head())
    print("\n")

