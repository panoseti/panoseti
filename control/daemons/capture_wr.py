#! /usr/bin/env python3

##############################################################
# Script for capturing White Rabbit data from the WR Switches
# and writing their associated values into the Redis database.
# All packet information is time stamped by the computer and 
# added to each set of values with a variable labeled as
# 'Computer_UTC'.
##############################################################
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from datetime import UTC, datetime
from pathlib import Path
from signal import SIGINT, signal

from utils import config_file, util
from utils.panoseti_snmp import wrs_snmp
from utils.redis_utils import redis_init

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
        print(f"We can't connect to WR-SWITCH({wrs.dev})!")
        print('************************************************')
    else:
        print('*****************WR-SWITCH SFP CHECK***********************')
        if(res == 0):
            print(f'WR-SWITCH({wrs.dev}) : No sfp transceivers detected!')
        else:
            failed = 0
            for i in range(len(res)):
                if(len(res[i]) != 0):
                    if(res[i] != SFP_PN1):
                        failed = 1
                        print(f'WR-SWITCH({wrs.dev}) : sfp{i+1:2d} is {res[i]:<16}[ FAIL ]')
                    else:
                        print(f'WR-SWITCH({wrs.dev}) : sfp{i+1:2d} is {res[i]:<16}[ PASS ]')
            if failed == 0:
                print(' ')
                print(f'WR-SWITCH({wrs.dev}) : sfp transceivers are checked!')
                print(' ')
            else:
                print(' ')
                print('Error : Please check the sfp transceivers!!')
                print(f'The part number of the sfp transceiver should be {SFP_PN1}')
                print(' ')

# check the link status
#
def wrsLinkStatusCheck(wrs):
    res = wrs.linkstatus()
    if(res == -1):
        print('********************Error***************************')
        print(f"We can't connect to WR-Endpoint({wrs.dev})!")
        print('****************************************************')
    else:
        print('*****************WR-SWITCH LINK CHECK***********************')
        if(res == 0):
            print(f'WR-SWITCH({wrs.dev}) : No sfp transceivers detected!')
        else:
            for i in range(len(res)):
                if res[i] == LINK_UP :
                    print(f'WR-SWITCH({wrs.dev}) : Port{i+1:2d} LINK_UP  ')
                else:
                    print(f'WR-SWITCH({wrs.dev}) : Port{i+1:2d} LINK_DOWN')
    print(' ')

# check the softpll status
#
def wrsSoftPLLCheck(wrs):
    res = wrs.pllstatus()
    if(res[0] == -1):
        print('********************Error***************************')
        print(f"We can't connect to WR-Endpoint({wrs.dev})!")
        print('****************************************************')
    else:
        print('***************WR-SWITCH SoftPLL CHECK**********************')
        if(res == SOFTPLL_LOCKED):
            print('WR-SWITCH({}) SoftPLL Status: {}'.format(wrs.dev, 'LOCKED'))
        elif(res == SOFTPLL_UNLOCKED):
            print('WR-SWITCH({}) SoftPLL Status: {}'.format(wrs.dev, 'UNLOCK'))
            print('Please Check 10MHz and 1PPS!!!')
        else:
            print('WR-SWITCH({}) SoftPLL Status: {}({})'.format(wrs.dev, 'WEIRD STATUS', res[0]))
            print('WEIRD STATUS! Please Check 10MHz and 1PPS!!!')
        print(' ')


# init redis and create wrs_snmp obj
#
def initialize():
    r = redis_init()
    wrs = wrs_snmp(SWITCHIP)
    return wrs, r

def main():
    script_dir = Path(__file__).resolve().parent
    os.environ['MIBDIRS']= f'+{script_dir!s}/capture_wr'
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
            r.hset(RKEY, f'Port{i+1:2d}_LINK', 1 if res[i] == LINK_UP else 0)
        # check softpll status
        res = wrs.pllstatus()
        r.hset(RKEY, 'SOFTPLL', 1 if res[0] == SOFTPLL_LOCKED else 0)
        print(datetime.now(UTC).replace(tzinfo=None))
        time.sleep(1)

if __name__ == "__main__":
    main()
