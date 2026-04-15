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
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Any

import redis
from influxdb import InfluxDBClient
from requests.exceptions import ConnectionError
from rich.console import Console

# Rich Logging Imports
from rich.logging import RichHandler

# Add control root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add telemetry subdirectory to path for local utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'capture_telemetry_service')))

from utils import config_file
from utils.redis_utils import get_casted_redis_value, redis_init

try:
    from archiver_utils import TelemetryConfigManager
except ImportError:
    print("WARNING: Could not import archiver_utils. Dynamic telemetry disabled.")
    TelemetryConfigManager = None

# --- CONFIGURATION VARIABLES
# InfluxDB database names
DB_PROD_NAME = 'metadata'
DB_DEV_NAME = 'dev_metadata'

# Retention Policies
RP_PROD_DEFAULT = 'autogen'  # Infinite (InfluxDB default)
RP_DEV_NAME = 'autogen_7d'  # We set the DB default to 7 days, so 'autogen' is 7 days
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

# --- LOGGING SETUP
# Configure Rich logging for pretty, structured output
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("storeInfluxDB")

# --- Code

# List of keys with the time stamp values
# - Track timestamps to avoid duplicate writes
# - Structure: {redis_key: last_seen_computer_utc}
key_timestamps: dict[str, str] = {}


def init_influx_clients() -> tuple[InfluxDBClient, InfluxDBClient]:
    """
    Robustly connects to InfluxDB (Prod and Dev) with retry logic.
    Blocks until connection succeeds.

    Returns:
        Tuple[InfluxDBClient, InfluxDBClient]: A tuple containing the production and development clients.
    """
    logger.info("Connecting to InfluxDB...")
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
                logger.warning(f"Could not apply retention policy for {RP_DEV_NAME}")
                pass
            except Exception:
                # Retention policy likely already exists
                pass

            logger.info("[bold green]✅ Connected to InfluxDB.[/]", extra={"markup": True})
            return client_prod, client_dev

        except (ConnectionError, Exception) as e:
            logger.warning(f"⚠️ InfluxDB not ready: {e}. Retrying in 5s...")
            time.sleep(5)


def determine_routing(
        rkey: str,
        client_prod: InfluxDBClient,
        client_dev: InfluxDBClient,
        telemetry_mgr: Any | None
) -> tuple[InfluxDBClient | None, str | None]:
    """
    Decides where a Redis Key should go based on Legacy Regex or Dynamic Config.

    Args:
        rkey (str): The Redis key to check.
        client_prod (InfluxDBClient): The production InfluxDB client.
        client_dev (InfluxDBClient): The development InfluxDB client.
        telemetry_mgr (Optional[Any]): The dynamic configuration manager.

    Returns:
        Tuple[Optional[InfluxDBClient], Optional[str]]: The target client and the datatype tag, or (None, None).
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


def extract_redis_payload(r: redis.Redis, rkey: str) -> dict[str, Any] | None:
    """
    Fetches hash from Redis, performs type casting, and deduplicates based on timestamp.

    Args:
        r (redis.Redis): The Redis client connection.
        rkey (str): The Redis key to extract.

    Returns:
        Optional[Dict[str, Any]]: Dict of data fields if new data exists, else None.
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

        for field_b, _val_b in raw_hash.items():
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
        logger.error(f"Error extracting payload for {rkey}: {e}")
        return None


def format_influx_point(key: str, data_fields: dict[str, Any], datatype: str) -> dict[str, Any] | None:
    """
    Formats a raw dictionary of fields into an InfluxDB JSON point structure.

    Args:
        key (str): The measurement name (Redis key).
        data_fields (Dict[str, Any]): The data fields to store.
        datatype (str): The datatype tag.

    Returns:
        Optional[Dict[str, Any]]: The formatted InfluxDB point or None if formatting fails.
    """
    try:
        ts_val = data_fields.get('Computer_UTC', time.time())

        # TIMEZONE HANDLING:
        # Force UTC conversion to prevent 'Naive Time' errors where InfluxDB assumes local time.
        # Strict ISO format with 'Z' suffix denotes UTC.
        t_obj = datetime.utcfromtimestamp(float(ts_val))
        t = t_obj.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        return {
            "measurement": key,
            "tags": {
                "observatory": OBSERVATORY,
                "datatype": datatype
            },
            "fields": data_fields,
            "time": t
        }
    except Exception as e:
        logger.error(f"Error formatting Influx point for {key}: {e}")
        return None


def process_redis_keys(
        r: redis.Redis,
        client_prod: InfluxDBClient,
        client_dev: InfluxDBClient,
        telemetry_mgr: Any | None
):
    """
    Main processing logic: Scans keys, routes them, extracts data, and prepares batch writes.
    Opt: Uses r.scan_iter() for non-blocking iteration and writes in batches.

    Args:
        r (redis.Redis): Redis client.
        client_prod (InfluxDBClient): Production InfluxDB client.
        client_dev (InfluxDBClient): Development InfluxDB client.
        telemetry_mgr (Optional[Any]): Config manager.
    """
    batch_prod = []
    batch_dev = []

    try:
        # Opt: Use scan_iter instead of keys('*') to prevent blocking Redis
        # as the key space grows. This is O(1) per call vs O(N).
        # We scan for ALL keys and let determine_routing filter them.
        for rkey_b in r.scan_iter(match='*'):
            rkey = rkey_b.decode('utf-8')

            # A. Route the update to either the Production or Development InfluxDB client
            target_client, datatype = determine_routing(rkey, client_prod, client_dev, telemetry_mgr)

            if not target_client:
                continue

            # B. Load the Redis data snapshot for rkey
            data_fields = extract_redis_payload(r, rkey)

            # C. Format and buffer the point if data exists
            if data_fields:
                point = format_influx_point(rkey, data_fields, datatype)
                if point:
                    if target_client == client_prod:
                        batch_prod.append(point)
                    else:
                        batch_dev.append(point)

        # D. Bulk Write Batches (This is an optimization that reduces InfluxDB operations)
        if batch_prod:
            client_prod.write_points(batch_prod)
            # logger.debug(f"Wrote {len(batch_prod)} points to PRODUCTION")

        if batch_dev:
            client_dev.write_points(batch_dev)
            # logger.debug(f"Wrote {len(batch_dev)} points to DEVELOPMENT")

    except redis.RedisError as e:
        logger.error(f"Redis connection lost during scan: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in process loop: {e}")


def main():
    logger.info("storeInfluxDB: Starting Dual-Database Archiver (Optimized)...")

    # 1. Initialize the Redis client and Prod/Dev InfluxDB clients
    try:
        r = redis_init()
        client_prod, client_dev = init_influx_clients()
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    telemetry_mgr = TelemetryConfigManager() if TelemetryConfigManager else None
    logger.info(f"Telemetery Manager Active: [bold]{telemetry_mgr is not None}[/]", extra={"markup": True})

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
            logger.error(f"Unexpected error in main loop: {e}")

        time.sleep(UPDATE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()