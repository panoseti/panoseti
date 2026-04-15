#! /usr/bin/env python3
##############################################################
# Verbose version of capture_hk2.py
# Logs each step to the console for debugging
##############################################################

import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from signal import SIGINT, signal
from sys import exit

import redis

# ===== Path to panoseti project root =====
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from capture_hk import metadata_status_monitor_utils as md_utils
from capture_hk.panosetiSIconvert import HKconvert

from utils.redis_utils import redis_init

# ===== CONFIGURATION =====
HOST = '0.0.0.0'
PORT = 60002
OBSERVATORY = "Gattini"   # e.g., "Winter", "Fern", "PTI", "Gattini"

BASE_DIR = "/mnt/data11/data/palomar/L0"
REMOTE_SERVER = "panoseti@132.239.146.24"
REMOTE_DIR  = "/home/www/current"
REMOTE_DIR2 = "www/current"

HKconv = HKconvert()
HKconv.changeUnits('V')
HKconv.changeUnits('A')

COUNTER = "\rPackets Captured So Far {}"

signed = [
    0,0,0,0,0, 0,0,0,0, 0,0,0,0,0, 0,0,0,
    1,0,0,0, 0,0,0,0, 0,0, 0,0,0,0
]

##############################################################
def get_true_detector_current(raw_detector_current_uA, detector_hv_volts):
    return raw_detector_current_uA - (abs(detector_hv_volts) / 0.499) / 1_000_000

def handler(sig, frame):
    print('\n? SIGINT or CTRL-C detected. Exiting gracefully.')
    exit(0)

def run_command(cmd, label):
    """Run a shell command and print status."""
    print(f"   ??  {label}: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"   ? {label} succeeded.")
        return True
    except subprocess.CalledProcessError:
        print(f"   ??  {label} FAILED.")
        return False

def append_to_daily_log(redis_set, boardName):
    """Append housekeeping record to local daily JSON log."""
    date_str = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d")
    log_dir = os.path.join(BASE_DIR, date_str, OBSERVATORY, "hk")
    log_file = os.path.join(log_dir, f"{boardName}.json")

    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(redis_set) + "\n")
        print(f"   ? Log appended to: {log_file}")
        return True
    except Exception as e:
        print(f"   ? Failed to write log {log_file}: {e}")
        return False

def send_latest_to_remote(redis_set, boardName):
    """Send latest record to remote 'current' folder."""
    tmpfile = f"/tmp/{boardName}_hk.json"
    try:
        with open(tmpfile, "w") as f:
            json.dump(redis_set, f, indent=2)
        print(f"   ? Temp file written: {tmpfile}")
    except Exception as e:
        print(f"   ? Failed to create temp file: {e}")
        return

    scp_cmd = f"scp -q {tmpfile} {REMOTE_SERVER}:{REMOTE_DIR}/"
    chmod_cmd = f"ssh {REMOTE_SERVER} 'chmod 644 {REMOTE_DIR2}/{os.path.basename(tmpfile)}'"

    run_command(scp_cmd, "SCP upload")
    run_command(chmod_cmd, "Remote chmod")

    try:
        os.remove(tmpfile)
        print("   ? Temp file removed.")
    except FileNotFoundError:
        pass

##############################################################
def storeInRedis(packet, r: redis.Redis):
    """Decode HK packet, store to Redis, log to file, and sync latest."""
    array = []
    startUp = 0

    if int.from_bytes(packet[0:1], byteorder='little') != 0x20:
        return False
    if int.from_bytes(packet[1:2], byteorder='little') == 0xaa:
        startUp = 1

    for i, sign in zip(range(2, len(packet), 2), signed, strict=False):
        array.append(int.from_bytes(packet[i:i+2], byteorder='little', signed=sign))

    boardName = "QUABO_" + str(array[0])
    print(f"\n? Received packet from {boardName}")

    redis_set = {
        'Computer_UTC': time.time(),
        'BOARDLOC': array[0],
        'HVMON0': HKconv.convertValue('HVMON0', array[1]),
        'HVMON1': HKconv.convertValue('HVMON1', array[2]),
        'HVMON2': HKconv.convertValue('HVMON2', array[3]),
        'HVMON3': HKconv.convertValue('HVMON3', array[4]),
        'HVIMON0': HKconv.convertValue('HVIMON0', array[5]),
        'HVIMON1': HKconv.convertValue('HVIMON1', array[6]),
        'HVIMON2': HKconv.convertValue('HVIMON2', array[7]),
        'HVIMON3': HKconv.convertValue('HVIMON3', array[8]),
        'RAWHVMON': HKconv.convertValue('RAWHVMON', -array[9]),
        'V12MON': HKconv.convertValue('V12MON', array[10]),
        'V18MON': HKconv.convertValue('V18MON', array[11]),
        'V33MON': HKconv.convertValue('V33MON', array[12]),
        'V37MON': HKconv.convertValue('V37MON', array[13]),
        'I10MON': HKconv.convertValue('I10MON', array[14]),
        'I18MON': HKconv.convertValue('I18MON', array[15]),
        'I33MON': HKconv.convertValue('I33MON', array[16]),
        'TEMP1': HKconv.convertValue('TEMP1', array[17]),
        'TEMP2': HKconv.convertValue('TEMP2', array[18]),
        'VCCINT': HKconv.convertValue('VCCINT', array[19]),
        'VCCAUX': HKconv.convertValue('VCCAUX', array[20]),
        'UID': f'0x{array[24]:04x}{array[23]:04x}{array[22]:04x}{array[21]:04x}',
        'SHUTTER_STATUS': array[25]&0x01,
        'LIGHT_SENSOR_STATUS': (array[25]&0x02) >> 1,
        'PCBREV_N': ((array[25]&0xFF00) >> 8) & 0x01,
        'FWTIME': f'0x{array[28]:04x}{array[27]:04x}',
        'FWVER': bytes.fromhex(f'{array[30]:04x}{array[29]:04x}').decode("ASCII"),
        'StartUp': startUp,
        'AGG_STATUS_MSG': "",
        'AGG_STATUS_LEVEL': 0
    }

    for x in range(4):
        redis_set[f'DETR{x}_CURR'] = get_true_detector_current(
            redis_set[f'HVIMON{x}'], redis_set[f'HVMON{x}'])

    try:
        md_utils.write_status("housekeeping", boardName, redis_set)
        for key, val in redis_set.items():
            r.hset(boardName, key, val)
        print(f"   ? Redis update succeeded for {boardName}")
    except Exception as e:
        print(f"   ? Redis update failed: {e}")

    append_to_daily_log(redis_set, boardName)
    send_latest_to_remote(redis_set, boardName)

##############################################################
def initialize():
    print("? Initializing socket and Redis connection...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    r = redis_init()
    print("   ? Redis connection OK")
    return sock, r

##############################################################
signal(SIGINT, handler)

def main():
    sock, r = initialize()
    print(f"? Running and listening on UDP port {PORT}")
    sock.bind((HOST, PORT))
    num = 0
    while True:
        packet = sock.recvfrom(64)
        num += 1
        storeInRedis(packet[0], r)
        print(COUNTER.format(num), end='')

if __name__ == "__main__":
    main()
