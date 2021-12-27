import os
import netsnmp
import redis
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

    return r


def main():
    r = initialize()
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
        

        r.hset(RKEY, 'SYSTIME', datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
        for i in range(len(res)):
            tmp = bytes.decode(res[i]).replace(' ','') 					#convert bytes to str, and replace the 'space' at the end
            r.hset(RKEY, 'Port%2d_LINK'%(i+1), 1 if tmp == LINK_UP else 0)

        print(datetime.utcnow())
        time.sleep(1)

if __name__ == "__main__":
    main()