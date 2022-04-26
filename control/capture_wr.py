#! /usr/bin/env python3

##############################################################
# Script for capturing White Rabbit data from the WR Switches
# and writing their associated values into the Redis database.
# All packet information is time stamped by the computer and 
# added to each set of values with a variable labeled as
# 'Computer_UTC'.
##############################################################
import os
import netsnmp
import redis
import time
from signal import signal, SIGINT
import time
from datetime import datetime
from redis_utils import *
import config_file, util

from panoseti_snmp import wrs_snmp

# wrs status
LINK_DOWN   =   '1'
LINK_UP     =   '2'
SFP_PN0     =   'PS-FB-TX1310'
SFP_PN1     =   'PS-FB-RX1310'
SOFTPLL_LOCKED      =   '1'
SOFTPLL_UNLOCKED    =   '2'

SWITCHIP    =   util.get_wr_ip_addr(config_file.get_obs_config())
RKEY        =   f'WRSWITCH{""}'
OBSERVATORY =   'lick'

def handler(signal_recieved, frame):
    print('\nSIGINT or CTRL-C detected. Exiting')
    exit(0)
signal(SIGINT, handler)

#------------------------------------------------------------#
# check the PN of SFP transceivers
#
def wrsSFPCheck(wrs):
    res = wrs.sfppn()
    if(res == -1):
        print('************************************************')
        print("We can't connect to WR-SWITCH(%s)!"%(wrs.dev))
        print('************************************************')
    else:
        print('*****************WR-SWITCH SFP CHECK***********************')
        if(res == 0):
            print('WR-SWITCH(%s) : No sfp transceivers detected!' %(wrs.dev))
        else:
            failed = 0
            for i in range(len(res)):
                if(len(res[i]) != 0):
                    if(res[i] != SFP_PN1):
                        failed = 1
                        print('WR-SWITCH(%s) : sfp%2d is %-16s[ FAIL ]' %(wrs.dev, i+1, res[i]))
                    else:
                        print('WR-SWITCH(%s) : sfp%2d is %-16s[ PASS ]' %(wrs.dev, i+1, res[i]))
            if failed == 0:
                print(' ')
                print('WR-SWITCH(%s) : sfp transceivers are checked!' % (wrs.dev))
                print(' ')
            else:
                print(' ')
                print('Error : Please check the sfp transceivers!!')
                print('The part number of the sfp transceiver should be %s'%(SFP_PN1))
                print(' ')

# check the link status
#
def wrsLinkStatusCheck(wrs):
    res = wrs.linkstatus()
    if(res == -1):
        print('********************Error***************************')
        print("We can't connect to WR-Endpoint(%s)!"%(wrs.dev))
        print('****************************************************')
    else:
        print('*****************WR-SWITCH LINK CHECK***********************')
        if(res == 0):
            print('WR-SWITCH(%s) : No sfp transceivers detected!' %(wrs.dev))
        else:
            for i in range(len(res)):
                if res[i] == LINK_UP :
                    print('WR-SWITCH(%s) : Port%2d LINK_UP  ' %(wrs.dev, i+1))
                else:
                    print('WR-SWITCH(%s) : Port%2d LINK_DOWN' %(wrs.dev, i+1))
    print(' ')

# check the softpll status
#
def wrsSoftPLLCheck(wrs):
    res = wrs.pllstatus()
    if(res[0] == -1):
        print('********************Error***************************')
        print("We can't connect to WR-Endpoint(%s)!"%(wrs.dev))
        print('****************************************************')
    else:
        print('***************WR-SWITCH SoftPLL CHECK**********************')
        if(res == SOFTPLL_LOCKED):
            print('WR-SWITCH(%s) SoftPLL Status: %s'%(wrs.dev, 'LOCKED'))
        elif(res == SOFTPLL_UNLOCKED):
            print('WR-SWITCH(%s) SoftPLL Status: %s'%(wrs.dev, 'UNLOCK'))
            print('Please Check 10MHz and 1PPS!!!')
        else:
            print('WR-SWITCH(%s) SoftPLL Status: %s(%s)'%(wrs.dev, 'WEIRD STATUS', res[0]))
            print('WEIRD STATUS! Please Check 10MHz and 1PPS!!!')
        print(' ')


# init redis and create wrs_snmp obj
#
def initialize():
    r = redis_init()
    wrs = wrs_snmp(SWITCHIP)
    return wrs, r

def main():
    wrs, r = initialize()
    # check the current status one time, including sfpPN, link status and softpll status,
    # and print the info out
    wrsSFPCheck(wrs)
    wrsLinkStatusCheck(wrs)
    wrsSoftPLLCheck(wrs)

    # then check link status and softpll status once a second,
    # and write the status into redis
    while(True):
        r.hset(RKEY, 'Computer_UTC', time.time())
        # check link status
        res = wrs.linkstatus()
        for i in range(len(res)):
            r.hset(RKEY, 'Port%2d_LINK'%(i+1), 1 if res[i] == LINK_UP else 0)
        # check softpll status
        res = wrs.pllstatus()
        r.hset(RKEY, 'SOFTPLL', 1 if res[0] == SOFTPLL_LOCKED else 0)
        print(datetime.utcnow())
        time.sleep(1)

if __name__ == "__main__":
    main()
