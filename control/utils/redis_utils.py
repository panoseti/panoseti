##############################################################
# Utility functions for communicating with redis and sending 
# commands to redis databases
##############################################################
import re
from typing import Any

import redis


def redis_init() -> redis.Redis:
    return redis.Redis(host='localhost', port=6379, db=0)


def store_in_redis(r: redis.Redis, rkey: bytes | str, rkey_fields: dict) -> None:
    """
    Writes every field from rkey_fields into the hashset stored at rkey
    in the Redis database represented by the object r.
    """
    rk = rkey.decode("utf-8") if isinstance(rkey, bytes) else str(rkey)
    for field, value in rkey_fields.items():
        r.hset(rk, field, value)


def get_updated_redis_keys(r: redis.Redis, key_timestamps: dict) -> list[str]:
    # r.keys returns a list of bytes or a list of awaitables depending on the redis client version
    # The sync client returns a list of bytes.
    keys_raw = r.keys('*')
    if not isinstance(keys_raw, list):
        return []
    
    avaliable_keys = [key.decode("utf-8") if isinstance(key, bytes) else str(key) for key in keys_raw]
    list_of_updates = []
    for key in avaliable_keys:
        try:
            compUTC = r.hget(key, 'Computer_UTC')
            if compUTC is None:
                continue
            # compUTC is bytes from hget in sync client
            compUTC_str = compUTC.decode("utf-8") if isinstance(compUTC, bytes) else str(compUTC)
            if key in key_timestamps and key_timestamps[key] == compUTC_str:
                continue
            list_of_updates.append(key)
        except redis.ResponseError:
            pass
    return list_of_updates


def get_casted_redis_value(r: redis.Redis, rkey: bytes | str, field: bytes | str) -> Any:
    """Returns val = r.hget(rkey, field) casted to int, float, or string
     as follows:
        1. int, if val has the form X where X.isnumeric(),
        2. float, if val has the form (-)X.Y where X.isnumeric() and Y.isnumeric(),
        3. string otherwise.
    """
    val_raw = None
    # Checks if val exists in the provided Redis database.
    try:
        rk = rkey.decode("utf-8") if isinstance(rkey, bytes) else str(rkey)
        fld = field.decode("utf-8") if isinstance(field, bytes) else str(field)
        val_raw = r.hget(rk, fld)
    except redis.RedisError as rerr:
        msg = "redis_utils.py: A Redis error occurred: {0}."
        print(msg.format(rerr))
        pass
    
    if val_raw is not None:
        val: str = val_raw.decode('utf-8') if isinstance(val_raw, bytes) else str(val_raw)
        # Checks if val has the form X, with X numeric.
        if val.isnumeric() or (len(val) > 0 and val[0] == '-' and val[1:].isnumeric()):
            return int(val)
        # Checks if val has the form (-)X.Y, with X and Y numeric.
        pattern = re.compile(r"^-*([0-9]+)\.([0-9]+?)(?:[eE]-?\+?([0-9]+))?$")
        match = pattern.match(val)
        if match and match.group(1).isnumeric() and match.group(2).isnumeric() \
                and (match.group(3) is None or match.group(3).isnumeric()):
            return float(val)
        return val
