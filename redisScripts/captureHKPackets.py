import socket
import redis
from influxdb import InfluxDBClient
from signal import signal, SIGINT
from sys import exit
from datetime import datetime

HOST = '0.0.0.0'
PORT = 60002

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

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

r = redis.Redis(host='localhost', port=6379, db=0)

client = InfluxDBClient('localhost', 8086, 'root', 'root', 'metadata')
client.create_database('metadata')

def handler(signal_recieved, frame):
    print('\nSIGINT or CTRL-C detected. Exiting')
    sock.close()
    exit(0)
    
def getUID(intArr):
    return intArr[0] + (intArr[1] << 16) + (intArr[2] << 32) + (intArr[3] << 48)
    
def storeInRedisandInflux(packet):
    array = []
    startUp = 0
    
    if int.from_bytes(packet[0:1], byteorder='little') != 0x20:
        return False
    if int.from_bytes(packet[1:2], byteorder='little') == 0xaa:
        startUp = 1
        
    for i, sign in zip(range(2,len(packet), 2), signed):
        array.append(int.from_bytes(packet[i:i+2], byteorder='little', signed=sign))
        
    json_body = [
        {
            "measurement": "Quabo{0}".format(array[0]),
            "tags": {
                "observatory": "lick",
                "datatype": "housekeeping"
            },
            "time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fields":{
                'BOARDLOC': array[0],
                'HVMON0': array[1],
                'HVMON1': array[2],
                'HVMON2': array[3],
                'HVMON3': array[4],

                'HVIMON0': array[5],
                'HVIMON1': array[6],
                'HVIMON2': array[7],
                'HVIMON3': array[8],

                'RAWHVMON': array[9],

                'V12MON': array[10],
                'V18MON': array[11],
                'V33MON': array[12],
                'V37MON': array[13],

                'I10MON': array[14],
                'I18MON': array[15],
                'I33MON': array[16],

                'TEMP1': array[17],
                'TEMP2': array[18],

                'VCCINT': array[19],
                'VCCAUX': array[20],

                'UID': getUID(array[21:25]),

                'SHUTTER_STATUS': array[25]&0x01,
                'LIGHT_SENSOR_STATUS': (array[25]&0x02) >> 1,

                'FWID0': array[27] + array[28]*0x10000,
                'FWID1': array[29] + array[30]*0x10000           
            }
        }
    ]
    client.write_points(json_body)
    
    boardName = array[0]
    r.hset(boardName, 'SYSTIME', json_body[0]['time'])
    r.hset(boardName,'BOARDLOC', boardName)
    r.hset(boardName, 'HVMON0', array[1]) #array[1]*1.22/1000)
    r.hset(boardName, 'HVMON1', array[2]) #array[2]*1.22/1000)
    r.hset(boardName, 'HVMON2', array[3]) #array[3]*1.22/1000)
    r.hset(boardName, 'HVMON3', array[4]) #array[4]*1.22/1000)
    
    r.hset(boardName, 'HVIMON0', array[5]) #(65535-array[5])*38.1/1000000)
    r.hset(boardName, 'HVIMON1', array[6]) #(65535-array[6])*38.1/1000000)
    r.hset(boardName, 'HVIMON2', array[7]) #(65535-array[7])*38.1/1000000)
    r.hset(boardName, 'HVIMON3', array[8]) #(65535-array[8])*38.1/1000000)
    
    r.hset(boardName, 'RAWHVMON', array[9]) #array[9]*1.22/1000)
    
    r.hset(boardName, 'V12MON', array[10]) #array[10]*19.07/1000000000)
    r.hset(boardName, 'V18MON', array[11]) #array[11]*19.07/1000000000)
    r.hset(boardName, 'V33MON', array[12]) #array[12]*38.1/1000000000)
    r.hset(boardName, 'V37MON', array[13]) #array[13]*38.1/1000000000)
    
    r.hset(boardName, 'I10MON', array[14]) #array[14]*182/1000000)
    r.hset(boardName, 'I18MON', array[15]) #array[15]*37.8/1000000)
    r.hset(boardName, 'I33MON', array[16]) #array[16]*37.8/1000000)
    
    r.hset(boardName, 'TEMP1', array[17]) #array[17]*0.0625)
    r.hset(boardName, 'TEMP2', array[18]) #array[18]*0.0625)
    
    r.hset(boardName, 'VCCINT', array[19]) #array[19]*3/65536)
    r.hset(boardName, 'VCCAUX', array[20]) #array[20]*3/65536)
     
    r.hset(boardName, 'UID', json_body[0]['fields']['UID'])
    
    r.hset(boardName, 'SHUTTER_STATUS', json_body[0]['fields']['SHUTTER_STATUS'])
    r.hset(boardName, 'LIGHT_SENSOR_STATUS', json_body[0]['fields']['LIGHT_SENSOR_STATUS'])
    
    r.hset(boardName, 'FWID0', json_body[0]['fields']['FWID0'])
    r.hset(boardName, 'FWID1', json_body[0]['fields']['FWID1'])

    r.hset(boardName,'StartUp', startUp)

    r.hset('UPDATED', boardName, "1")

    
    
signal(SIGINT, handler)

print('Running')
sock.bind((HOST,PORT))
num = 0
while(True):
    packet = sock.recvfrom(64)
    num += 1
    storeInRedisandInflux(packet[0])
    print(COUNTER.format(num), end='')
