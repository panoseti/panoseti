import redis

def redis_init():
    return redis.Redis(host='localhost', port=6379, db=0)

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