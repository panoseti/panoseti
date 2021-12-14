import redis

r = redis.Redis(host='localhost', port=6379, db=0)

redis_key = "WRSWITCH"
print("{")
print("    " + redis_key + ": {")
redis_value = r.hgetall(redis_key)
for k in redis_value.keys():
    print("        {0}: {1}".format(k.decode("utf-8"), redis_value[k].decode("utf-8")))
print("}")