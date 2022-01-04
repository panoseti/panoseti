from io import FileIO
import redis
import json
import sys
import time

file_ptr = None
r = redis.Redis(host='localhost', port=6379, db=0)
#List of keys with the time stamp values
key_timestamps = {}

def get_updated_redis_keys(key_timestamps):
    avaliable_keys = [key.decode("utf-8") for key in r.keys('*')]
    list_of_updates = []
    for key in avaliable_keys:
        try:
            compUTC = r.hget(key, 'Computer_UTC')
            if compUTC == None:
                continue
            if key in key_timestamps and key_timestamps[key] == compUTC.decode("utf-8"):
                continue
            list_of_updates.append(key)
        except redis.ResponseError:
            pass
    return list_of_updates
    

def write_redis_keys(file_ptr:FileIO, redis_keys:dict, key_timestamps:dict):
    for rkey in redis_keys:
        redis_value = r.hgetall(rkey)
        value_dict = { k.decode('utf-8'): redis_value[k].decode('utf-8') for k in redis_value.keys() }
        json.dump({rkey: value_dict}, file_ptr)
        key_timestamps[rkey] = value_dict['Computer_UTC']

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a file for output")
        exit(0)
    elif len(sys.argv) > 2:
        print("Too many command line arguments")
        exit(0)
    file_ptr = open(sys.argv[1], "w+")
    while True:
        write_redis_keys(file_ptr, get_updated_redis_keys(key_timestamps), key_timestamps)
        time.sleep(1)