#! /usr/bin/env python3
##############################################################
# Populates new data from redis into the influxDB database.
# Script stores all sets which contains the key for the 
# computer timestamp 'Computer_UTC'. All sets where this value 
# is absent is ignored. The set is stored as a new entry in the
# database 'metadata' in the measurement associated with each 
# redis set.
##############################################################
import os
import sys
import time
import re
import redis
from datetime import datetime
from influxdb import InfluxDBClient

# --- PATH SETUP ---
# Add control root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add telemetry subdirectory to path for local utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'capture_telemetry_service')))

from utils import config_file
from utils.redis_utils import redis_init, get_casted_redis_value
# Import the helper we just created
try:
    from archiver_utils import TelemetryConfigManager
except ImportError:
    print("WARNING: Could not import archiver_utils. Dynamic telemetry disabled.")
    TelemetryConfigManager = None

# --- CONFIGURATION VARIABLES ---
# InfluxDB database names
DB_PROD_NAME = 'metadata'
DB_DEV_NAME  = 'dev_metadata'

# Retention Policies
RP_PROD_DEFAULT = 'autogen'   # Infinite (InfluxDB default)
RP_DEV_DEFAULT  = 'autogen'   # We set the DB default to 7 days, so 'autogen' is 7 days
DEV_RETENTION_DURATION = '7d'

# Seconds between consecutive snapshots of Redis
UPDATE_INTERVAL_SECONDS = 1.0

# Discover current observatory name
OBSERVATORY = config_file.get_obs_config()["name"]

# Static whitelist of redis keys to include in InfluxDB snapshots.
DATATYPE_FORMAT = {
    'housekeeping': re.compile("QUABO_\\d*"),
    'GPS': re.compile("GPS.*"),
    'whiterabbit': re.compile("WRSWITCH.*"),
    'outlet': re.compile("WPS.*"),
    # 'ublox_f9t': re.compile("UBLOX_ZED-F9T_.*"), # not implemented
    'mount': re.compile("MOUNT_.*"),
    'power': re.compile("POWER_.*"),
    'weather': re.compile("WEATHER.*"),
}
# List of keys with the time stamp values
# - Track timestamps to avoid duplicate writes
# - Structure: {redis_key: last_seen_computer_utc}
key_timestamps = {}


def init_influx_clients():
    """
    Initialize two separate clients to enforce isolation.
    """
    # 1. Production Client
    client_prod = InfluxDBClient('localhost', 8086, 'root', 'root', DB_PROD_NAME)
    client_prod.create_database(DB_PROD_NAME)

    # 2. Development Client
    client_dev = InfluxDBClient('localhost', 8086, 'root', 'root', DB_DEV_NAME)
    client_dev.create_database(DB_DEV_NAME)

    # Enforce Retention on Dev DB
    # We alter the 'autogen' policy of the dev DB to be 7 days.
    # This ensures that even if users forget to specify RP, it defaults to deletion.
    try:
        client_dev.create_retention_policy('autogen_7d', DEV_RETENTION_DURATION, "1", default=True)
    except Exception:
        # Policy might already exist, safe to ignore
        pass

    return client_prod, client_dev


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
def write_to_influx(client:InfluxDBClient, key:str, data_fields:dict, datatype:str):
    """
    Creates the json body and write the data to influxDB.
        - Generic write function for either client (prod or dev).
    """
    try:
        # Robust timestamp extraction
        ts_val = data_fields.get('Computer_UTC', time.time())
        t = datetime.utcfromtimestamp(float(ts_val)).isoformat()

        json_body = [{
            "measurement": key,
            "tags": {
                "observatory": OBSERVATORY,
                "datatype": datatype
            },
            "fields": data_fields,
            "time": t
        }]

        client.write_points(json_body)
    except Exception as e:
        print(f"Error writing to Influx ({key}): {e}")


def write_redis_to_influx(client:InfluxDBClient, r:redis.Redis, redis_keys:list, key_timestamps:dict):
    print("Updating keys:", redis_keys)
    for rkey in redis_keys:
        data_fields = dict()
        for key in r.hkeys(rkey):
            val = get_casted_redis_value(r, rkey, key)
            if (val is not None) and (val != ""):
                data_fields[key.decode('utf-8')] = val
            else:
                msg = f"storeInfluxDB.py: No data in ({rkey}, {key.decode('utf-8')}): {repr(val)}!"
                msg += "\n Aborting influx write..."
                continue
        write_influx(client, rkey, data_fields, get_datatype(rkey))
        key_timestamps[rkey] = data_fields['Computer_UTC']


def main():
    # r, client = influx_init()
    # key_timestamps = {}
    # while True:
    #     write_redis_to_influx(client, r, get_updated_redis_keys(r, key_timestamps), key_timestamps)
    #     time.sleep(1)

    print("Starting storeInfluxDB")

    r = redis_init()
    client_prod, client_dev = init_influx_clients()

    # Initialize the specific Telemetry Service utility
    if TelemetryConfigManager:
        telemetry_mgr = TelemetryConfigManager()
    else:
        telemetry_mgr = None

    while True:
        # Reload telemetry service config if changed
        if telemetry_mgr:
            telemetry_mgr.reload()

        # Safe key iteration
        try:
            # Using keys() is standard for this script, though scan_iter is safer for massive DBs
            all_keys = [k.decode('utf-8') for k in r.keys('*')]
        except redis.RedisError:
            time.sleep(UPDATE_INTERVAL_SECONDS)
            continue

        # Process all Redis Keys
        for rkey in all_keys:
            target_client = None
            datatype_tag = None

            # --- ROUTING LOGIC ---

            # 1. Priority: Legacy Regex (Existing Daemons)
            # This guarantees backward compatibility
            for dtype, regex in DATATYPE_FORMAT.items():
                if regex.match(rkey):
                    datatype_tag = dtype
                    target_client = client_prod
                    break

            # 2. Priority: Dynamic Telemetry Service
            if not target_client and telemetry_mgr:
                dtype, mode = telemetry_mgr.match_key(rkey)
                if dtype:
                    datatype_tag = dtype
                    if mode == 'production':
                        target_client = client_prod
                    else:
                        target_client = client_dev  # Experimental -> Dev DB

            # If no route found, skip this key
            if not target_client:
                continue

            # --- DATA EXTRACTION & WRITE ---
            try:
                # Optimistic check: Has the timestamp changed?
                # This saves us from pulling the whole hash if nothing happened.
                comp_utc_raw = r.hget(rkey, 'Computer_UTC')
                if not comp_utc_raw:
                    continue

                comp_utc = comp_utc_raw.decode('utf-8')

                # Deduplication check
                if key_timestamps.get(rkey) == comp_utc:
                    continue

                # Fetch Payload
                raw_hash = r.hgetall(rkey)
                data_fields = {}

                for field_b, val_b in raw_hash.items():
                    field = field_b.decode('utf-8')
                    # Use central utility to cast strings back to numbers
                    val = get_casted_redis_value(r, rkey, field)
                    if val is not None and val != "":
                        data_fields[field] = val

                if data_fields:
                    write_to_influx(target_client, rkey, data_fields, datatype_tag)
                    key_timestamps[rkey] = comp_utc

            except Exception as e:
                # Log but keep looping
                # print(f"Skipping {rkey}: {e}")
                pass

            time.sleep(UPDATE_INTERVAL_SECONDS)

        if __name__ == "__main__":
            main()


if __name__ == "__main__":
    main()
