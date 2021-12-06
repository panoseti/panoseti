#! /usr/bin/env python

# Configure and reboot quabos
# options:
# --show        show summary of config file (list of domes and modules)
# --ping        ping selected quabos
# --reboot      reboot selected quabos
# --loads       load silver firmware in selected quabos
# --loadg       load gold firmware in selected quabos
#               DEPRECATED - dangerous
# based on matlab/initq.m, startq*.py

firmware_silver = 'quabo_0116C_23CBEAFB.bin'
firmware_gold = 'quabo_GOLD_23BD5DA4.bin'

import util, config_file, sys, os, quabo_driver
from panoseti_tftp import tftpw

def usage():
    print('''usage:
--show        show list of domes/modules/quabos
--ping        ping selected quabos
--reboot      reboot selected quabos
--loads       load silver firmware in selected quabos
''')
    sys.exit()

# print summary of obs config file
#
def show_config(obs_config):
    for dome in obs_config['domes']:
        print('dome %s'%dome['num'])
        for module in dome['modules']:
            module_num = module['num']
            ip_addr = module['ip_addr']
            print('   module %s'%module_num)
            print('      Mobo serial#: %s'%module['mobo_serialno'])
            for i in range(4):
                quabo_num = module_num*4+i
                quabo_ip = util.quabo_ip_addr(ip_addr, i)
                print('      quabo %d'%quabo_num)
                print('         IP addr: %s'%quabo_ip)

def do_reboot(modules, quabo_uids):
    # need to reboot quabos in order 0..3
    # could do this in parallel across modules; for now, do it serially
    #
    for module in modules:
        for i in range(4):
            if not config_file.is_quabo_alive(module, quabo_uids, i):
                continue;
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            x = tftpw(ip_addr)
            x.reboot()

            # wait for a housekeeping packet
            #
            quabo = quabo_driver.QUABO(ip_addr)
            while True:
                if quabo.read_hk_packet():
                    break
            quabo.close()

def do_loads(modules, quabo_uids):
    for module in modules:
        for i in range(4):
            if not config_file.is_quabo_alive(module, quabo_uids, i):
                continue;
            ip_addr = util.quabo_ip_addr(module['ip_addr'], i)
            x = tftpw(ip_addr)
            x.put_bin_file(firmware_silver)

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
            show_config(obs_config)
            sys.exit()
        elif argv[i] == '--reboot':
            nops += 1
            op = 'reboot'
        elif argv[i] == '--loads':
            nops += 1
            op = 'loads'
        elif argv[i] == '--ping':
            nops += 1
            op = 'ping'
        else:
            print('bad arg: %s'%argv[i])
            usage()
        i += 1

    if (nops == 0):
        print('no op specified')
        usage()
    if (nops > 1):
        print('must specify a single op')
        usage()

    modules = config_file.get_modules(obs_config)
    quabo_uids = config_file.get_quabo_uids()
    if op == 'reboot':
        do_reboot(modules, quabo_uids)
    elif op == 'loads':
        do_loads(modules, quabo_uids)
    elif op == 'ping':
        do_ping(modules)
