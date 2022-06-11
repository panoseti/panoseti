#! /usr/bin/env python3

##############################################################
# Script for capturing Housekeeping data from the quabos
# and writing their associated values into the Redis database.
# All packet information is time stamped by the computer and
# and added to each set of values with a variable labeled as
# 'Computer_UTC'.
##############################################################
import socket
import redis
from signal import signal, SIGINT
from sys import exit
import time
from datetime import datetime
from redis_utils import *

from panosetiSIconvert import HKconvert
HKconv = HKconvert()
HKconv.changeUnits('V')
HKconv.changeUnits('A')

HOST = '0.0.0.0'
PORT = 60002
OBSERVATORY = "lick"

COUNTER = "\rPackets Captured So Far {}"

signed = [
    0,                        # BOARDLOC
    0, 0, 0, 0,          # HVMON (0 to -80V)
    0, 0, 0, 0, # HVIMON ((65535-HVIMON) * 38.1nA) (0 to 2.5mA)
    0,                        # RAWHVMON (0 to -80V)
    0,                      # V12MON (19.07uV/LSB) (1.2V supply)
    0,                      # V18MON (38.14uV/LSB) (1.8V supply)
    0,                      # V33MON (76.20uV/LSB) (3.3V supply)
    0,                      # V37MON (76.20uV/LSB) (3.7V supply)
    0,                      # I10MON (182uA/LSB) (1.0V supply)
    0,                      # I18MON (37.8uA/LSB) (1.8V supply)
    0,                      # I33MON (37.8uA/LSB) (3.3V supply)
    1,                        # TEMP1 (0.0625*N)
    0,                      # TEMP2 (N/130.04-273.15)
    0,                      # VCCINT (N*3/65536)
    0,                      # VCCAUX (N*3/65536)
    0,0,0,0,                    # UID
    0,                        # SHUTTER, LIGHT_SENSOR STATUS, and PCBREV_N
    0,                        # unused
    0,0,0,0                     # FWID0 and FWID1
]

def handler(signal_recieved, frame):
    print('\nSIGINT or CTRL-C detected. Exiting')
    exit(0)
    
def getUID(intArr):
    return intArr[0] + (intArr[1] << 16) + (intArr[2] << 32) + (intArr[3] << 48)
    
def storeInRedis(packet, r:redis.Redis):
    array = []
    startUp = 0
    
    if int.from_bytes(packet[0:1], byteorder='little') != 0x20:
        return False
    if int.from_bytes(packet[1:2], byteorder='little') == 0xaa:
        startUp = 1

    # Reads the housekeeping bytes with offsets 2 through 63 two at a time, converts the
    # 16-bit number to an integer with a sign determined by the corresponding entry in signed,
    # and appends the result to array.
    # The byte with offset 2 <= n <= 63 in the HK packet is obtained as follows:
    #              array[(n - 2) // 2] & 0x00FF, if n is even
    #       (array[(n - 2) // 2] & 0xFF00) >> 8, if n is odd.
    for i, sign in zip(range(2,len(packet), 2), signed):
        array.append(int.from_bytes(packet[i:i+2], byteorder='little', signed=sign))
        
    boardName = "QUABO_" + str(array[0])
    
    redis_set = {
        'Computer_UTC': time.time(),#datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        'BOARDLOC': array[0],
        'HVMON0': '{0:0.5g}'.format(HKconv.convertValue('HVMON0', array[1])),
        'HVMON1': '{0:0.5g}'.format(HKconv.convertValue('HVMON1', array[2])),
        'HVMON2': '{0:0.5g}'.format(HKconv.convertValue('HVMON2', array[3])),
        'HVMON3': '{0:0.5g}'.format(HKconv.convertValue('HVMON3', array[4])),

        'HVIMON0': '{0:0.5g}'.format(HKconv.convertValue('HVIMON0', array[5])),
        'HVIMON1': '{0:0.5g}'.format(HKconv.convertValue('HVIMON1', array[6])),
        'HVIMON2': '{0:0.5g}'.format(HKconv.convertValue('HVIMON2', array[7])),
        'HVIMON3': '{0:0.5g}'.format(HKconv.convertValue('HVIMON3', array[8])),

        'RAWHVMON': '{0:0.5g}'.format(HKconv.convertValue('RAWHVMON', -array[9])),

        'V12MON': '{0:0.5g}'.format(HKconv.convertValue('V12MON', array[10])),
        'V18MON': '{0:0.5g}'.format(HKconv.convertValue('V18MON', array[11])),
        'V33MON': '{0:0.5g}'.format(HKconv.convertValue('V33MON', array[12])),
        'V37MON': '{0:0.5g}'.format(HKconv.convertValue('V37MON', array[13])),

        'I10MON': '{0:0.5g}'.format(HKconv.convertValue('I10MON', array[14])),
        'I18MON': '{0:0.5g}'.format(HKconv.convertValue('I18MON', array[15])),
        'I33MON': '{0:0.5g}'.format(HKconv.convertValue('I33MON', array[16])),

        'TEMP1': '{0:0.5g}'.format(HKconv.convertValue('TEMP1', array[17])),
        'TEMP2': '{0:0.5g}'.format(HKconv.convertValue('TEMP2', array[18])),

        'VCCINT': '{0:0.5g}'.format(HKconv.convertValue('VCCINT', array[19])),
        'VCCAUX': '{0:0.5g}'.format(HKconv.convertValue('VCCAUX', array[20])),

        'UID': '0x{0:04x}{1:04x}{2:04x}{3:04x}'.format(array[24],array[23],array[22],array[21]),

        'SHUTTER_STATUS': array[25]&0x01,
        'LIGHT_SENSOR_STATUS': (array[25]&0x02) >> 1,

        # PCBrev_n represents the quabo version. If 0, the quabo is BGA version; if 1, the qubao is QFP version
        # Bit 0 in the byte with offset 53.
        'PCBREV_N': ((array[25]&0xFF00) >> 8) & 0x01,

        'FWTIME': '0x{0:04x}{1:04x}'.format(array[28],array[27]),
        'FWVER': bytes.fromhex('{0:04x}{1:04x}'.format(array[30],array[29])).decode("ASCII"),
        
        'StartUp': startUp
    }

    for key in redis_set.keys():
        r.hset(boardName, key, redis_set[key])

def initialize():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    r = redis_init()
    return sock, r
    
    
    
signal(SIGINT, handler)

def main():
    sock, r = initialize()
    print('Running')
    sock.bind((HOST,PORT))
    num = 0
    while(True):
        packet = sock.recvfrom(64)
        num += 1
        storeInRedis(packet[0], r)
        print(COUNTER.format(num), end='')

if __name__ == "__main__":
    main()
