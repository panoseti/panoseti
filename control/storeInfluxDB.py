#! /usr/bin/env python3
##############################################################
# Populates new data from redis into the influxDB database.
# Script stores all sets which contains the key for the 
# computer timestamp 'Computer_UTC'. All sets where this value 
# is absent is ignored. The set is stored as a new entry in the
# database 'metadata' in the measurement associated with each 
# redis set.
##############################################################
from os import write
from influxdb import InfluxDBClient
import redis
import time
import re
from redis_utils import *

#UTC_OFFSET = 7*3600 #ns
UTC_OFFSET = 0
TIMEFORMAT = "%Y-%m-%dT%H:%M:%SZ"
OBSERVATORY = 'test'
DATATYPE_FORMAT = {'housekeeping': re.compile("Quabo_[0-9]*"),
    'GPS': re.compile("GPS.*"),
    'whiterabbit': re.compile("WRSWITCH.*")}
#List of keys with the time stamp values
key_timestamps = {}


def influx_init():
    r = redis_init()
    client = InfluxDBClient('panoseti-influxdb', 8086, 'root', 'root', 'metadata')
    client.create_database('metadata')

    return r, client

def get_datatype(redis_key):
    for key in DATATYPE_FORMAT.keys():
        if DATATYPE_FORMAT[key].match(redis_key) is not None:
            return key
    return "None"

# Create the json body and write the data to influxDB
def write_influx(client:InfluxDBClient, key:str, data_fields:dict, datatype:str):
    t0 = float(data_fields['Computer_UTC']) + UTC_OFFSET
    t1 = time.localtime(t0)
    t = time.strftime(TIMEFORMAT, t1)
    json_body = [
        {
            "measurement": key,
            "tags": {
                "observatory": OBSERVATORY,
                "datatype": datatype
            },
            #"time": data_fields['Computer_UTC'],
            "time": t,
            "fields": data_fields
        }
    ]
    client.write_points(json_body)

def write_redis_to_influx(client:InfluxDBClient, r:redis.Redis, redis_keys:list, key_timestamps:dict):
    for rkey in redis_keys:
        redis_value = r.hgetall(rkey)
        data_fields = { k.decode('utf-8'): redis_value[k].decode('utf-8') for k in redis_value.keys() }
        write_influx(client, rkey, data_fields, get_datatype(rkey))
        key_timestamps[rkey] = data_fields['Computer_UTC']

def main():
    r, client = influx_init()
    key_timestamps = {}
    while True:
        write_redis_to_influx(client, r, get_updated_redis_keys(r, key_timestamps), key_timestamps)
        time.sleep(1)

if __name__ == "__main__":
    main()
