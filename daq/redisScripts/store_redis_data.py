import redis
import json
import sys
import time

file_ptr = None
r = redis.Redis(host='localhost', port=6379, db=0)

def get_updated_redis_keys(update_key):
    update_values = r.hgetall(update_key)
    list_of_updates = []
    for k in update_values.keys():
        if update_values[k] == b'1':
            list_of_updates.append(k.decode("utf-8"))
    return list_of_updates
    

def write_redis_key(redis_keys):
    for rkey in redis_keys:
        redis_value = r.hgetall(rkey)
        value_dict = { k.decode('utf-8'): redis_value[k].decode('utf-8') for k in redis_value.keys() }
        json.dump({rkey: value_dict}, file_ptr)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a file for output")
        exit(0)
    elif len(sys.argv) > 2:
        print("Too many command line arguments")
        exit(0)
    file_ptr = open(sys.argv[1], "w+")
    while True:
        write_redis_key(get_updated_redis_keys("UPDATED"))
        file_ptr.write("\n\n")
        time.sleep(1)