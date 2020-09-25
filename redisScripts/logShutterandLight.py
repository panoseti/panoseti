import socket
import redis
from signal import signal, SIGINT
from sys import exit
from datetime import datetime

r = redis.Redis(host='localhost', port=6379, db=0)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

f = open('STATUS.log', 'a+')

def handler(signal_recieved, frame):
    print('\nSIGINT or CTRL-C detected. Exiting')
    sock.close()
    exit(0)

signal(SIGINT, handler)

print('Running\n')

while (True):
    updated = r.hgetall("UPDATED")
    for key in updated:
        if int(updated[key]):
            output = "BOARDLOC="+str(r.hget(key,"BOARDLOC"))[2:-1]
            output += " SYSTIME="+str(r.hget(key,"SYSTIME"))[2:-1]
            output += " SHUTTER_STATUS="+ str(r.hget(key,"SHUTTER_STATUS"))[2:-1]
            output += " LIGHT_SENSOR_STATUS="+str(r.hget(key,"LIGHT_SENSOR_STATUS"))[2:-1]+'\n'
            f.write(output)
            print(output)
            r.hset("UPDATED", key, 0)

