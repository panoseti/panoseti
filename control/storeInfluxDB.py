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
from datetime import datetime
import re
import config_file
from redis_utils import *

OBSERVATORY = config_file.get_obs_config()["name"]
DATATYPE_FORMAT = {
    'housekeeping': re.compile("QUABO_\\d*"),
    'GPS': re.compile("GPS.*"),
    'whiterabbit': re.compile("WRSWITCH.*"),
    'outlet': re.compile("UPS_.*")
}
# List of keys with the time stamp values
key_timestamps = {}


def influx_init():
    r = redis_init()
    client = InfluxDBClient('localhost', 8086, 'root', 'root', 'metadata')
    client.create_database('metadata')

    return r, client


def get_datatype(redis_key):
    for key in DATATYPE_FORMAT.keys():
        if DATATYPE_FORMAT[key].match(redis_key) is not None:
            return key
    return "None"


# Create the json body and write the data to influxDB
def write_influx(client:InfluxDBClient, key:str, data_fields:dict, datatype:str):
    utc_timestamp = data_fields['Computer_UTC']
    #utc_time_obj = datetime.utcfromtimestamp(utc_timestamp)
    #t = utc_time_obj.isoformat()
    json_body = [
        {
            "measurement": key,
            "tags": {
                "observatory": OBSERVATORY,
                "datatype": datatype
            },
            "fields": data_fields,
            "time": utc_timestamp
        }
    ]
    client.write_points(json_body)


def write_redis_to_influx(client:InfluxDBClient, r:redis.Redis, redis_keys:list, key_timestamps:dict):
    for rkey in redis_keys:
        data_fields = dict()
        for key in r.hkeys(rkey):
            val = get_casted_redis_value(r, rkey, key)
            if val is not None:
                data_fields[key.decode('utf-8')] = val
            else:
                msg = f"storeInfluxDB.py: No data in ({rkey.decode('utf-8')}, {key.decode('utf-8')}!"
                msg += "\n Aborting influx write..."
                print(msg)
                continue
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
