import socket
import redis
from signal import signal, SIGINT
from sys import exit
from datetime import datetime

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
    0,                      # V18MON (19.07uV/LSB) (1.8V supply)
    0,                      # V33MON (38.10uV/LSB) (3.3V supply)
    0,                      # V37MON (38.10uV/LSB) (3.7V supply)
    0,                      # I10MON (182uA/LSB) (1.0V supply)
    0,                      # I18MON (37.8uA/LSB) (1.8V supply)
    0,                      # I33MON (37.8uA/LSB) (3.3V supply)
    1,                        # TEMP1 (0.0625*N)
    0,                      # TEMP2 (N/130.04-273.15)
    0,                      # VCCINT (N*3/65536)
    0,                      # VCCAUX (N*3/65536)
    0,0,0,0,                    # UID
    0,                        # SHUTTER and LIGHT_SENSOR STATUS
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
        
    for i, sign in zip(range(2,len(packet), 2), signed):
        array.append(int.from_bytes(packet[i:i+2], byteorder='little', signed=sign))
        
    boardName = "QUABO_" + str(array[0])
    
    redis_set = {
        'Computer_UTC': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        'BOARDLOC': array[0],
        'HVMON0': HKconv.convertValue('HVMON0', array[1]),
        'HVMON1': HKconv.convertValue('HVMON1', array[2]),
        'HVMON2': HKconv.convertValue('HVMON2', array[3]),
        'HVMON3': HKconv.convertValue('HVMON3', array[4]),

        'HVIMON0': HKconv.convertValue('HVIMON0', array[5]),
        'HVIMON1': HKconv.convertValue('HVIMON1', array[6]),
        'HVIMON2': HKconv.convertValue('HVIMON2', array[7]),
        'HVIMON3': HKconv.convertValue('HVIMON3', array[8]),

        'RAWHVMON': HKconv.convertValue('RAWHVMON', array[9]),

        'V12MON': HKconv.convertValue('V12MON', array[10]),
        'V18MON': HKconv.convertValue('V18MON', array[11]),
        'V33MON': HKconv.convertValue('V33MON', array[12]),
        'V37MON': HKconv.convertValue('V37MON', array[13]),

        'I10MON': HKconv.convertValue('I10MON', array[14]),
        'I18MON': HKconv.codwnvertValue('I18MON', array[15]),
        'I33MON': HKconv.convertValue('I33MON', array[16]),
        'VCCINT': HKconv.convertValue('VCCINT', array[19]),
        'VCCAUX': HKconv.convertValue('VCCAUX', array[20]),

        'UID': '0x{0:04x}{0:04x}{0:04x}{0:04x}'.format(array[24],array[23],array[22],array[21]),

        'SHUTTER_STATUS': array[25]&0x01,
        'LIGHT_SENSOR_STATUS': (array[25]&0x02) >> 1,

        'FWID0': array[27] + array[28]*0x10000,
        'FWID1': array[29] + array[30]*0x10000,
        
        'StartUp': startUp
    }

    for key in redis_set.keys():
        r.hset(boardName, key, redis_set[key])

def initialize():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    r = redis.Redis(host='localhost', port=6379, db=0)
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
