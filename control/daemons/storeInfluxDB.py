#! /usr/bin/env python3
##############################################################
# Populates new data from redis into the influxDB database.
# Script stores all sets which contains the key for the 
# computer timestamp 'Computer_UTC'. All sets where this value 
# is absent is ignored. The set is stored as a new entry in the
# database 'metadata' in the measurement associated with each 
# redis set.
#
# This script processes metadata added to Redis by two kinds of daemons:
#   Type A (static, via custom scripts for each metadata type):
#       - Scripts: capture_<metadata type>.py, where <metadata type> is a distinct class of metadata source.
#           Captures metadata streams with custom handling.
#       - Modification: Generally requires code changes on remote clients and servers, usually with accompanied git commits for code distribution.
#
#   Type B (dynamic, via the unified Telemetry Service for all metadata types):
#       - Script: capture_telemetry_service.py:
#           Runs the Telemetry gRPC Service for high-performance metadata streams from remote Linux machines.
#       - Modification: possible with client-only modifications or changes to capture_telemetry_service/telemetry_config.toml
##############################################################
import os
import sys
import time
import re
import redis
from datetime import datetime
from influxdb import InfluxDBClient
from typing import Dict, Optional, Tuple, Any, List
from requests.exceptions import ConnectionError

# Add control root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add telemetry subdirectory to path for local utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'capture_telemetry_service')))

from utils import config_file
from utils.redis_utils import redis_init, get_casted_redis_value
try:
    from archiver_utils import TelemetryConfigManager
except ImportError:
    print("WARNING: Could not import archiver_utils. Dynamic telemetry disabled.")
    TelemetryConfigManager = None

# --- CONFIGURATION VARIABLES
# InfluxDB database names
DB_PROD_NAME = 'metadata'
DB_DEV_NAME  = 'dev_metadata'

# Retention Policies
RP_PROD_DEFAULT = 'autogen'   # Infinite (InfluxDB default)
RP_DEV_NAME  = 'autogen_7d'   # We set the DB default to 7 days, so 'autogen' is 7 days
DEV_RETENTION_DURATION = '7d'

# Seconds between consecutive snapshots of Redis
UPDATE_INTERVAL_SECONDS = 1.0

# Discover current observatory name
OBSERVATORY = config_file.get_obs_config()["name"]

# --- Type A metadata definitions
# Static whitelist of Redis keys to include in InfluxDB snapshots.
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


# --- Code

# List of keys with the time stamp values
# - Track timestamps to avoid duplicate writes
# - Structure: {redis_key: last_seen_computer_utc}
key_timestamps = {}


def init_influx_clients() -> Tuple[InfluxDBClient, InfluxDBClient]:
    """
    Robustly connects to InfluxDB (Prod and Dev) with retry logic.
    Blocks until connection succeeds.
    """
    print("Connecting to InfluxDB...")
    while True:
        try:
            # 1. Production Client
            client_prod = InfluxDBClient('localhost', 8086, 'root', 'root', DB_PROD_NAME)
            client_prod.create_database(DB_PROD_NAME)

            # 2. Development Client
            client_dev = InfluxDBClient('localhost', 8086, 'root', 'root', DB_DEV_NAME)
            client_dev.create_database(DB_DEV_NAME)

            # Enforce Retention on Dev DB
            try:
                client_dev.create_retention_policy(RP_DEV_NAME, DEV_RETENTION_DURATION, "1", default=True)
            except ConnectionError:
                print(f"WARNING: Could not apply retention policy for {RP_DEV_NAME}")
                pass

            print("✅ Connected to InfluxDB.")
            return client_prod, client_dev

        except (ConnectionError, Exception) as e:
            print(f"⚠️ InfluxDB not ready: {e}. Retrying in 5s...")
            time.sleep(5)


def determine_routing(
        rkey: str,
        client_prod: InfluxDBClient,
        client_dev: InfluxDBClient,
        telemetry_mgr: Optional[Any]
) -> Tuple[Optional[InfluxDBClient], Optional[str]]:
    """
    Decides where a Redis Key should go based on Legacy Regex or Dynamic Config.

    Returns:
        (TargetClient, DatatypeTag) or (None, None) if no match found.
    """
    # 1. Type A: Legacy regex check (Priority 1 for backward compatibility)
    # This logic ensures we keep the original storeInfluxDB functionality.
    for dtype, regex in DATATYPE_FORMAT.items():
        if regex.match(rkey):
            return client_prod, dtype

    # 2. Type B: Dynamic telemetry check (Priority 2: new services interfacing with the gRPC Telemetry Service)
    if telemetry_mgr:
        dtype, mode = telemetry_mgr.match_key(rkey)
        if dtype:
            target = client_prod if mode == 'production' else client_dev
            return target, dtype

    return None, None


def extract_redis_payload(r: redis.Redis, rkey: str) -> Optional[Dict[str, Any]]:
    """
    Fetches hash from Redis, performs type casting, and deduplicates based on timestamp.

    Returns:
        Dict of data fields if new data exists, else None.
    """
    try:
        # Optimistic Timestamp Check
        comp_utc_raw = r.hget(rkey, 'Computer_UTC')
        if not comp_utc_raw:
            return None

        comp_utc = comp_utc_raw.decode('utf-8')

        # Deduplication: Has this timestamp been processed?
        if key_timestamps.get(rkey) == comp_utc:
            return None

        # Fetch Full Hash
        raw_hash = r.hgetall(rkey)
        data_fields = {}

        for field_b, val_b in raw_hash.items():
            field = field_b.decode('utf-8')
            # do robust type casting (str -> int/float)
            val = get_casted_redis_value(r, rkey, field)
            if val is not None and val != "":
                data_fields[field] = val

        if not data_fields:
            return None

        # Update cache after successful extraction
        # Note: We return the payload, caller handles the side-effect update or we do it here.
        # Doing it here assumes success, which is optimistic but efficient.
        key_timestamps[rkey] = comp_utc
        return data_fields

    except Exception as e:
        print(f"Error for {rkey}: {e}")
        return None


def write_to_influx(client: InfluxDBClient, key: str, data_fields: Dict[str, Any], datatype: str):
    """
    Formats the payload and writes a single point to the specified InfluxDB client.
    """
    try:
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
        # Uncomment for high-verbosity debugging:
        # print(f"DEBUG: Wrote {key} to {client._database}")

    except Exception as e:
        print(f"Error writing to Influx ({key}): {e}")


def process_redis_keys(
        r: redis.Redis,
        client_prod: InfluxDBClient,
        client_dev: InfluxDBClient,
        telemetry_mgr: Optional[Any]
):
    """
    Main processing logic: Scans keys, routes them, extracts data, and writes to DB.
    """
    try:
        # Get all keys (Safe decode)
        all_keys = [k.decode('utf-8') for k in r.keys('*')]
    except redis.RedisError:
        print("Redis connection lost during scan.")
        return

    for rkey in all_keys:
        # A. Route the update to either the Production or Development InfluxDB client
        target_client, datatype = determine_routing(rkey, client_prod, client_dev, telemetry_mgr)

        if not target_client:
            continue

        # B. Load the Redis data snapshot for rkey
        data_fields = extract_redis_payload(r, rkey)

        # C. Write the Redis data snapshot to InfluxDB.
        if data_fields:
            write_to_influx(target_client, rkey, data_fields, datatype)


def main():
    print("storeInfluxDB: Starting Dual-Database Archiver (Refactored)...")

    # 1. Initialize the Redis client and Prod/Dev InfluxDB clients
    try:
        r = redis_init()
        client_prod, client_dev = init_influx_clients()
    except Exception as e:
        print(f"CRITICAL: Initialization failed: {e}")
        return

    telemetry_mgr = TelemetryConfigManager() if TelemetryConfigManager else None
    print(f"Telemetery Manager Active: {telemetry_mgr is not None}")

    # 2. Update loop: continue forever until killed
    while True:
        # Dynamic Telemetry Service configuration reload
        #   - Enables the storeInfluxDB script to dynamically accept new types of metadata from the Telemetry Service.
        #   - The purpose of this feature is to enable rapid iteration when creating or modifying metadata scripts.
        if telemetry_mgr:
            telemetry_mgr.reload()

        # Process batch
        try:
            process_redis_keys(r, client_prod, client_dev, telemetry_mgr)
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")

        time.sleep(UPDATE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()