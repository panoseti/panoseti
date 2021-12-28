from influxdb import InfluxDBClient
import redis
import time
import re

OBSERVATORY = 'Lick'
DATATYPE_FORMAT = {'housekeeping': re.compile("Quabo_[0-9]*"),
    'GPS': re.compile("GPS.*"),
    'whiterabbit': re.compile("WRSWITCH.*")}


def influx_init():
    r = redis.Redis(host='localhost', port=6379, db=0)
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
            "time": data_fields['SYSTIME'],
            "fields": data_fields
        }
    ]
    client.write_points(json_body)

def main():
    r, client = influx_init()
    key_timestamps = {}
    while True:
        avaliable_keys = [key.decode("utf-8") for key in r.keys('*')]
        for key in avaliable_keys:
            try:
                systime = r.hget(key, 'SYSTIME')
                if systime == None:
                    continue
                if key in key_timestamps and key_timestamps[key] == systime:
                    continue
                
                redis_set = r.hgetall(key)
                data_fields = {}
                for rkey in redis_set:
                    data_fields[rkey.decode("utf-8")] = redis_set[rkey].decode("utf-8")
                
                write_influx(client, key, data_fields, get_datatype(key))
                key_timestamps[key] = systime
            except redis.ResponseError:
                pass
        time.sleep(1)

if __name__ == "__main__":
    main()