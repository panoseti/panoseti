##############################################################
# Utility functions for communicating with redis and sending 
# commands to redis databases
##############################################################
import re
import redis


def redis_init():
    return redis.Redis(host='localhost', port=6379, db=0)


def store_in_redis(r: redis.Redis, rkey: [bytes, str], rkey_fields: dict):
    """
    Writes every field from rkey_fields into the hashset stored at rkey
    in the Redis database represented by the object r.
    """
    for field, value in rkey_fields.items():
        r.hset(rkey, field, value)


def get_updated_redis_keys(r:redis.Redis, key_timestamps:dict):
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


def get_casted_redis_value(r:redis.Redis, rkey: [bytes, str], field: [bytes, str]):
    """Returns val = r.hget(rkey, field) casted to int, float, or string
     as follows:
        1. int, if val has the form X where X.isnumeric(),
        2. float, if val has the form (-)X.Y where X.isnumeric() and Y.isnumeric(),
        3. string otherwise.
    """
    val = None
    # Checks if val exists in the provided Redis database.
    try:
        val = r.hget(rkey, field)
    except redis.RedisError as rerr:
        msg = "redis_utils.py: A Redis error occurred: {0}."
        print(msg.format(rerr))
        pass
    if val is not None:
        val = val.decode('utf-8')
        # Checks if val has the form X, with X numeric.
        if val.isnumeric() or (len(val) > 0 and val[0] == '-' and val[1:].isnumeric()):
            return int(val)
        # Checks if val has the form (-)X.Y, with X and Y numeric.
        pattern = re.compile("^-*([0-9]+)\.([0-9]+?)(?:[eE]-?\+?([0-9]+))?$")
        match = pattern.match(val)
        if match and match.group(1).isnumeric() and match.group(2).isnumeric() \
                and (match.group(3) is None or match.group(3).isnumeric()):
            return float(val)
        return val
