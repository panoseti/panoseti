#! /usr/bin/env python3

"""
Script for capturing metadata from each ethernet outlet
and storing it in the Redis database.
"""
from datetime import datetime
import time
from signal import signal, SIGINT
from sys import exit

import power
import redis_utils

# Time between updates.
UPDATE_INTERVAL = 1


def handler(signal_received, frame):
    print('\nSIGINT or CTRL-C detected. Exiting')
    exit(0)


def get_ups_packet():
    """Creates a dictionary of values to write into Redis."""
    power_status = "ON" if power.quabo_power_query() else "OFF"
    rkey_fields = {
        'Computer_UTC': time.time(),
        "POWER": power_status
    }
    return rkey_fields


def main():
    r = redis_utils.redis_init()
    print("capture_power: {0}\n\tRunning...".format(datetime.now()))
    while True:
        rkey = "UPS_0" # TODO: change config files to allow for multiple power outlets
        redis_utils.store_in_redis(r, rkey, get_ups_packet())
        time.sleep(UPDATE_INTERVAL)


signal(SIGINT, handler)
if __name__ == "__main__":
    main()

