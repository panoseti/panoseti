#! /usr/bin/env python3

##############################################################
# Store data from redis database into panoseti metdata json 
# files. Script stores all sets which contains the key for the 
# computer timestamp 'Computer_UTC'. All sets where this value 
# is absent is ignored. The set is then stored in to a json 
# format separated by the characters '\n\n'.
# As pertained in the panoseti metdata json format specifications.
##############################################################
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
from typing import TextIO

from utils.redis_utils import get_updated_redis_keys, redis_init

file_ptr = None
r = redis_init()
#List of keys with the time stamp values
key_timestamps: dict[str, str] = {}    

def write_redis_keys(file_ptr: TextIO, redis_keys: list[str], key_timestamps: dict[str, str]) -> None:
    for rkey in redis_keys:
        redis_value = r.hgetall(rkey)
        if not isinstance(redis_value, dict):
            continue
        value_dict = { (k.decode('utf-8') if isinstance(k, bytes) else str(k)): (v.decode('utf-8') if isinstance(v, bytes) else str(v)) for k, v in redis_value.items() }
        json.dump({rkey: value_dict}, file_ptr)
        file_ptr.write("\n\n")
        key_timestamps[rkey] = value_dict.get('Computer_UTC', '')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a file for output")
        exit(0)
    elif len(sys.argv) > 2:
        print("Too many command line arguments")
        exit(0)
    with open(sys.argv[1], "w+") as file_ptr:
        while True:
            write_redis_keys(file_ptr, get_updated_redis_keys(r, key_timestamps), key_timestamps)
            time.sleep(1)
