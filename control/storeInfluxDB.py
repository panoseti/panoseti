from os import write
from influxdb import InfluxDBClient
import redis
import time
import re
from redis_utils import *

OBSERVATORY = 'Lick'
DATATYPE_FORMAT = {'housekeeping': re.compile("Quabo_[0-9]*"),
    'GPS': re.compile("GPS.*"),
    'whiterabbit': re.compile("WRSWITCH.*")}
#List of keys with the time stamp values
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
    json_body = [
        {
            "measurement": key,
            "tags": {
                "observatory": OBSERVATORY,
                "datatype": datatype
            },
            "time": data_fields['Computer_UTC'],
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