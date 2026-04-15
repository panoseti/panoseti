#! /usr/bin/env python3

# open or close shutter, which is controlled by quabo1.
# if the ip is not specified, we will open the shutter on all of the used modules.

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from argparse import ArgumentParser

from driver import quabo_driver
from utils import config_file, util

# check the ip address
# shutter controller is connected to quabo1
# 
if not os.path.exists('logs'):
        os.makedirs('logs')
logfile = 'logs/shutter.log'
util.create_logger(logfile, 'PANOSETI.Shutter', 'a')
logger = logging.getLogger('PANOSETI.Shutter')

def ip_check(ip):
    ip_str = ip.split('.')
    if(int(ip_str[3])%4==1):
        return 0
    else:
        return -1
    
# shutter operation
#
def shutterop(ip, port, op):
    # ip: ip address of quabo
    # port: port used for communicating with Quabo.
    # op: 0--open the shutter
    #     1--close the shutter
    if(op):
        opstr = 'close'
    else:
        opstr = 'open'
    logger.info('%s shutter on %s:%d'%(opstr,ip, port))
    quabo = quabo_driver.QUABO(ip, port)
    quabo.shutter_new(op)

def main():
    logger.info('************************************')
    parser = ArgumentParser(description="Usage for openning/closing shutter.")
    parser.add_argument("--ip",type=str, dest="ip",help="ip address of the quabo")
    parser.add_argument("--port",type=int, dest="port", default=60000, help="port used for communicating with Quabo.")
    parser.add_argument("--open", dest="open",action="store_true", help="open the shutter")
    parser.add_argument("--close", dest="close",action="store_true", help="close the shutter")
    opts = parser.parse_args()
    if(opts.open):
        op = 0
    if(opts.close):
        op = 1
    if(opts.ip):
        shutterop(opts.ip, opts.port, op)
        return
    obs_config = config_file.get_obs_config()
    network_config = config_file.get_network_config()
    for dome in obs_config['domes']:
        for m in dome['modules']:
            ip = config_file.quabo_ip_addr(m['ip_addr'],1)
            ip_ports = util.get_quabo_ip_port(m['ip_addr'], 1, network_config)
            real_ip = ip_ports['ip_addr']
            real_port = ip_ports['cmd_port']
            logger.debug('Quabo IP: %s'%real_ip)
            logger.debug('Real IP: %d'%real_port)
            shutterop(real_ip, real_port, op)

if __name__ == "__main__":
    main()