import os
import netsnmp
import redis
from influxdb import InfluxDBClient
import time
from signal import signal, SIGINT
from datetime import datetime


LINK_DOWN   =   '1'
LINK_UP     =   '2'
SFP_PN0     =   'PS-FB-TX1310'
SFP_PN1     =   'PS-FB-RX1310'
SWITCHIP    =   '192.168.1.254'
RKEY        =   f'WRSWITCH{""}'
OBSERVATORY =   'lick'

def handler(signal_recieved, frame):
    print('\nSIGINT or CTRL-C detected. Exiting')
    exit(0)
signal(SIGINT, handler)

def initialize():
    r = redis.Redis(host='localhost', port=6379, db=0)

    client = InfluxDBClient('localhost', 8086, 'root', 'root', 'metadata')
    client.create_database('metadata')

    os.environ['MIBDIRS']='+./'

    print('Help Information:')
    print('wrs_sfp      : get the sfp transceivers information on wr-switch')
    print('wrs_link     : get the link status of each port on wr-switch')
    check_flag = 0
    oid = netsnmp.Varbind('WR-SWITCH-MIB::wrsPortStatusSfpPN')
    try:
        res = netsnmp.snmpwalk(oid, Version=2, DestHost=SWITCHIP,Community='public')
    except:
        print('************************************************')
        print("We can't connect to WR-SWITCH(%s)!"%(SWITCHIP))
        print('************************************************')
        exit(0)

    print('*****************WR-SWITCH SFP CHECK***********************')

    if(res == None or len(res)==0):
        print('WR-SWITCH(%s) : No sfp transceivers detected!' %(SWITCHIP))
        exit(0)

    for i in range(len(res)):
        if len(res[i]) != 0:
            sfp_tmp = bytes.decode(res[i]).replace(' ','') 					#convert bytes to str, and replace the 'space' at the end
            if sfp_tmp != SFP_PN0 and sfp_tmp != SFP_PN1 :
                check_flag = 1
                print('WR-SWITCH(%s) : sfp%2d is %-16s[ FAIL ]' %(SWITCHIP, i+1, sfp_tmp))
            else:
                print('WR-SWITCH(%s) : sfp%2d is %-16s[ PASS ]' %(SWITCHIP, i+1, sfp_tmp))
    if check_flag == 0:
        print(' ')
        print('WR-SWITCH(%s) : sfp transceivers are checked!' % (SWITCHIP))
        print(' ')
    else:
        print(' ')
        print('Error : Please check the sfp transceivers!!')
        print(' ')

    return r, client


def main():
    r, client = initialize()
    while True:
        oid = netsnmp.Varbind('WR-SWITCH-MIB::wrsPortStatusLink')
        try:
            res = netsnmp.snmpwalk(oid, Version=2, DestHost=SWITCHIP, Community='public')
        except:
            print('********************Error***************************')
            print("We can't connect to WR-Endpoint(%s)!"%(SWITCHIP))
            print('****************************************************')
            exit(0)

        #print('*****************WR-SWITCH LINK CHECK***********************')
        if(len(res)==0):
            print('WR-SWITCH(%s) : No sfp transceivers detected!' %(SWITCHIP))
            exit(0)

        json_body = [
            {
                "measurement": "WRSwitch",
                "tags": {
                    "observatory": OBSERVATORY,
                    "datatype": "whiterabbit"
                },
                "time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "fields":{
                }
            }
        ]
        
        for i in range(len(res)):
            tmp = bytes.decode(res[i]).replace(' ','') 					#convert bytes to str, and replace the 'space' at the end
            if tmp == LINK_UP :
                r.hset(RKEY, 'Port%2d_LINK'%(i+1), 1)
                json_body[0]['fields']['Port%2d_LINK'%(i+1)] = 1
                #print('WR-SWITCH(%s) : Port%2d LINK_UP  ' %(SWITCHIP, i+1))
            else:
                r.hset(RKEY, 'Port%2d_LINK'%(i+1), 0)
                json_body[0]['fields']['Port%2d_LINK'%(i+1)] = 0
                #print('WR-SWITCH(%s) : Port%2d LINK_DOWN' %(SWITCHIP, i+1))
        
        client.write_points(json_body)

        r.hset('UPDATED', RKEY, 1)
        print(datetime.utcnow())
        time.sleep(1)

if __name__ == "__main__":
    main()