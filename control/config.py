#! /usr/bin/env python3

# Initialize for (one or more) observing runs
# options:
# --show            show list of domes/modules/quabos
# --ping            ping quabos
# --reboot          reboot quabos
# --loads           load silver firmware in quabos
# --init_daq_nodes  copy software to daq nodes
# --redis_daemons   start daemons to populate Redis with HK/GPS/WR data
#
# see matlab/initq.m, startq*.py

firmware_silver_qfp = 'quabo_0200_264489B3.bin'
firmware_silver_bga = 'quabo_0201_2644962F.bin'
firmware_gold = 'quabo_GOLD_23BD5DA4.bin'

import sys, os, subprocess
import util, config_file, quabo_driver, file_xfer
from panoseti_tftp import tftpw

def usage():
    print('''usage:
--show              show list of domes/modules/quabos
--ping              ping quabos
--reboot            reboot quabos
--loads             load silver firmware in quabos
--init_daq_nodes    copy software to daq nodes
--redis_daemons        start daemons to populate Redis with HK/GPS/WR data
''')
    sys.exit()

# print summary of obs and daq config files
#
def show_config(obs_config, quabo_uids):
    for dome in obs_config['domes']:
        print('dome %s'%dome['num'])
        for module in dome['modules']:
            module_id = module['id']
            ip_addr = module['ip_addr']
            print('   module ID %d'%module_id)
            print('      Mobo serial#: %s'%module['mobo_serialno'])
            for i in range(4):
                quabo_ip = util.quabo_ip_addr(ip_addr, i)
                print('      quabo %d'%i)
                print('         IP addr: %s'%quabo_ip)
    print("This node's IP addr: %s"%util.local_ip())
    config_file.show_daq_assignments(quabo_uids)

def do_reboot(modules, quabo_uids):
    # need to reboot quabos in order 0..3
    # could do this in parallel across modules; for now, do it serially
    #
    for module in modules:
        for i in range(4):
            if not util.is_quabo_alive(module, quabo_uids, i):
                continue
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            print('rebooting quabo at %s'%ip_addr)
            x = tftpw(ip_addr)
            x.reboot()
            print('waiting for HK packet from %s'%ip_addr)

            # wait for a housekeeping packet
            #
            quabo = quabo_driver.QUABO(ip_addr)
            while True:
                if quabo.read_hk_packet():
                    break
            quabo.close()
            print('rebooted quabo at %s'%ip_addr)

def do_loads(modules, quabo_uids):
    for module in modules:
        for i in range(4):
            if not util.is_quabo_alive(module, quabo_uids, i):
                continue
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            if util.is_quabo_old_version(module, i):
                fw = firmware_silver_qfp
            else:
                fw = firmware_silver_bga
            x = tftpw(ip_addr)
            print('loading %s into %s'%(fw, ip_addr))
            x.put_bin_file(fw)

def do_loadg(modules):
    print("not supported")
    #x.put_bin_file(firmware_gold, 0x0)

def do_ping(modules):
    for module in modules:
        for i in range(4):
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            if util.ping(ip_addr):
                print("pinged %s"%ip_addr)
            else:
                print("can't ping %s"%ip_addr)

if __name__ == "__main__":
    argv = sys.argv
    nops = 0
    obs_config = config_file.get_obs_config()
    i = 1
    while i < len(argv):
        if argv[i] == '--show':
            nops += 1
            op = 'show'
        elif argv[i] == '--reboot':
            nops += 1
            op = 'reboot'
        elif argv[i] == '--loads':
            nops += 1
            op = 'loads'
        elif argv[i] == '--ping':
            nops += 1
            op = 'ping'
        elif argv[i] == '--init_daq_nodes':
            nops += 1
            op = 'init_daq_nodes'
        elif argv[i] == '--redis_daemons':
            nops += 1
            op = 'redis_daemons'
        else:
            print('bad arg: %s'%argv[i])
            usage()
        i += 1

    if nops == 0:
        usage()
    if nops > 1:
        print('must specify a single op')
        usage()

    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    daq_config = config_file.get_daq_config()
    config_file.associate(daq_config, quabo_uids)
    if op == 'reboot':
        if util.local_ip() != util.default_hk_dest:
            print("You can only reboot quabos from %s"%(util.default_hk_dest))
            sys.exit()
        do_reboot(modules, quabo_uids)
    elif op == 'loads':
        do_loads(modules, quabo_uids)
    elif op == 'ping':
        do_ping(modules)
    elif op == 'init_daq_nodes':
        daq_config = config_file.get_daq_config()
        file_xfer.copy_hashpipe(daq_config)
    elif op == 'redis_daemons':
        util.start_redis_daemons()
    elif op == 'show':
        show_config(obs_config, quabo_uids)
        util.show_redis_daemons()
