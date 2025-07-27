#! /usr/bin/env python3

# open or close shutter, which is controlled by quabo1.
# if the ip is not specified, we will open the shutter on all of the used modules.

import sys, os, subprocess, time, datetime, json
from argparse import ArgumentParser
import util, file_xfer, quabo_driver
from panoseti_tftp import tftpw

sys.path.insert(0, '../util')
import pixel_coords
import config_file

# check the ip address
# shutter controller is connected to quabo1
# 
def ip_check(ip):
    ip_str = ip.split('.')
    if(int(ip_str[3])%4==1):
        return 0
    else:
        return -1
    
# shutter operation
#
def shutterop(ip,op):
    # ip: ip address of quabo
    # op: 0--open the shutter
    #     1--close the shutter
    if(op):
        opstr = 'close'
    else:
        opstr = 'open'
    print('%s shutter on %s'%(opstr,ip))
    quabo = quabo_driver.QUABO(ip)
    quabo.shutter_new(op)

def main():
    parser = ArgumentParser(description="Usage for openning/closing shutter.")
    parser.add_argument("--ip",type=str, dest="ip",help="ip address of the quabo")
    parser.add_argument("--open", dest="open",action="store_true", help="open the shutter")
    parser.add_argument("--close", dest="close",action="store_true", help="close the shutter")
    opts = parser.parse_args()
    if(opts.open):
        op = 0
    if(opts.close):
        op = 1
    if(opts.ip):
        if(ip_check(opts.ip)):
            raise Exception('Please make sure the IP address is correct.')
        shutterop(opts.ip,op)
        return
    obs_config = config_file.get_obs_config()
    for dome in obs_config['domes']:
        for m in dome['modules']:
            ip = config_file.quabo_ip_addr(m['ip_addr'],1)
            shutterop(ip,op)

if __name__ == "__main__":
    main()