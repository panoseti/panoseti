#! /usr/bin/env python3

# Periodically updates the high-voltage values in the quabos
# based on their latest temperature in the Redis database.
# See https://github.com/panoseti/panoseti/issues/47

import time
import redis

import redis_utils
from quabo_driver import QUABO
from panosetiSIconvert import HKconvert

# Seconds between updates.
UPDATE_INTERVAL = 10
# Detector nominal operating high voltage at 25 degrees C.
HV25C = ...  # TODO: Lookup actual value


def adjusted_voltage(temp: float):
    """Returns adjusted voltage. Assumes temp is in degrees Celsius."""
    return HV25C + (temp - 25) * 0.054


def update(r: redis.Redis):
    """Update voltage voltage values."""
    # TODO: implement
    try:
        # Set voltage
        pass
    except:
        pass


def main():
    r = redis_utils.redis_init()
    key_timestamps = {'TEMP1': None}  # This key does not exist?
    while True:
        # TODO
        time.sleep(UPDATE_INTERVAL)


main()
