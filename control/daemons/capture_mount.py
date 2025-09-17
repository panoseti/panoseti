#! /usr/bin/env python3

"""
Script for capturing metadata from each mount
and storing it in the Redis database.
"""
import os, sys
import socket
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
import time

# import power
from utils import redis_utils, config_file

# Time between updates.
UPDATE_INTERVAL = 1

def get_mount_fields(wps_dict):
    """Creates a dictionary of values to write into Redis."""
    try:
        # get mount metadata
        ...
    except Exception:
        print(f'capture_mount.py: Failed to query {wps_dict}."')
        raise
    ...
    rkey_fields = {
        'Computer_UTC': time.time(),
    }
    return rkey_fields


def get_mount_rkey(mount_key):
    """Returns the Redis key for the mount."""
    return mount_key.upper()

GATTINI_MOUNT_KEY = 'mount_gattini'
HOST = '0.0.0.0'
MOUNT_METADATA_PORT = 60005 # TBD???
PACKET_SIZE = 1024


def main():
    r = redis_utils.redis_init()
    # obs_config = config_file.get_obs_config()
    # mount_keys = [key for key in obs_config.keys() if 'wps' in key.lower()]
    # mount_keys = [GATTINI_MOUNT_KEY]
    print("capture_mount.py: Running...")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((HOST, MOUNT_METADATA_PORT))

        try:
            # Wait for a packet; this call blocks until a packet arrives.
            # It returns the data and the address (IP, port) of the sender.
            data_bytes, client_address = sock.recvfrom(PACKET_SIZE)

            print(f"\nReceived {len(data_bytes)} bytes from {client_address}")

            # 1. Strip padding and decode
            stripped_data = data_bytes.rstrip(b' ')
            json_str = stripped_data.decode('utf-8')

            # 2. Deserialize JSON to dictionary
            mount_data_dict = json.loads(json_str)
            print(f"Data: {mount_data_dict}")
            mount_key = mount_data_dict.get('mount_key', GATTINI_MOUNT_KEY)
            rkey = get_mount_rkey(mount_key)
            fields = get_mount_fields(mount_data_dict)
            print(f"{mount_key=}, {rkey=}, {fields=}")
            # redis_utils.store_in_redis(r, rkey, fields)

        except json.JSONDecodeError:
            print(f"Error decoding JSON from {client_address}. Packet may be malformed.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

