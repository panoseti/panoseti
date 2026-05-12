#! /usr/bin/env python3

##############################################################
# Store data from redis database into panoseti metdata json 
# files. Script stores all sets which contains the key for the 
# computer timestamp 'Computer_UTC'. All sets where this value 
# is absent is ignored. The set is then stored in to a json 
# format separated by the characters '\n\n'.
# As pertained in the panoseti metdata json format specifications.
##############################################################
import json
import signal
import sys
import time
from typing import TextIO

from control.utils.redis_utils import get_updated_redis_keys, redis_init

r = redis_init()
#List of keys with the time stamp values
key_timestamps: dict[str, str] = {}    
running = True

def write_redis_keys(file_ptr: TextIO, redis_keys: list[str], key_timestamps: dict[str, str]) -> None:
    for rkey in redis_keys:
        redis_value = r.hgetall(rkey)
        if not isinstance(redis_value, dict):
            continue
        value_dict = { (k.decode('utf-8') if isinstance(k, bytes) else str(k)): (v.decode('utf-8') if isinstance(v, bytes) else str(v)) for k, v in redis_value.items() }
        json.dump({rkey: value_dict}, file_ptr)
        file_ptr.write("\n\n")
        key_timestamps[rkey] = value_dict.get('Computer_UTC', '')

def signal_handler(sig, frame):
    global running
    print(f"\nSignal {sig} received. Flushing and exiting...")
    running = False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a file for output")
        exit(0)
    elif len(sys.argv) > 2:
        print("Too many command line arguments")
        exit(0)

    # Register for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    output_path = sys.argv[1]
    with open(output_path, "w+") as file_ptr:
        while running:
            write_redis_keys(file_ptr, get_updated_redis_keys(r, key_timestamps), key_timestamps)
            # Flush periodically to support real-time tailing
            file_ptr.flush()
            time.sleep(1)
        
        # Final flush on exit
        file_ptr.flush()
    print("Exited cleanly.")
