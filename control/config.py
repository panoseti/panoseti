#! /usr/bin/env python

# Configure and reboot quabos
# options:
# --show        show list of domes and modules
# --dome N      select dome
# --module N    select module
# --quabo ip_addr     select quabo
# --ping        ping selected quabos
# --reboot      reboot selected quabos
# --loads       load silver firmware in selected quabos
# --loadg       load gold firmware in selected quabos
#           DEPRECATED - dangerous
# based on matlab/initq.m, startq*.py

firmware_silver = 'quabo_0116C_23CBEAFB.bin'
firmware_gold = 'quabo_GOLD_23BD5DA4.bin'

import config_file, sys, os
from panoseti_tftp import tftpw

def usage():
    print('''usage:
--show        show list of domes/modules/quabos
--dome N      select dome
--module N    select module
--quabo ip_addr     select quabo
--ping        ping selected quabos
--reboot      reboot selected quabos
--loads       load silver firmware in selected quabos
''')
    sys.exit()

def show_quabos(obs_config):
    for dome in obs_config['domes']:
        print('dome %s'%dome['num'])
        for module in dome['modules']:
            module_num = module['num']
            ip_addr = module['ip_addr']
            print('   module %s'%module_num)
            print('      Mobo serial#: %s'%module['mobo_serialno'])
            for i in range(4):
                quabo_num = module_num*4+i
                quabo_ip = config_file.quabo_ip_addr(ip_addr, i)
                print('      quabo %d'%quabo_num)
                print('         IP addr: %s'%quabo_ip)

def do_op(quabo_ip_addrs, op):
    for ip_addr in quabo_ip_addrs:
        print(op, ip_addr)
        x = tftpw(ip_addr)
        if op == 'reboot':
            x.reboot()
        elif op == 'loads':
            x.put_bin_file(firmware_silver)
        elif op == 'loadg':
            print("not supported");
            #x.put_bin_file(firmware_gold, 0x0)
        elif op == 'ping':
            ret = os.system('ping -c 1 -w 1 -q %s > /dev/null 2>&1'%quabo['ip_addr'])
            if ret == 0:
                print('%s responds to ping'%quabo['ip_addr'])
            else:
                print('%s does not respond to ping'%quabo['ip_addr'])
            

if __name__ == "__main__":
    argv = sys.argv
    nops = 0
    nsel = 0
    dome = -1
    module = -1
    quabo_ip_addr = None
    obs_config = config_file.get_obs_config()
    i = 1
    while i < len(argv):
        if argv[i] == '--show':
            show_quabos(obs_config)
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
        elif argv[i] == '--dome':
            nsel += 1
            i += 1
            dome = int(argv[i])
        elif argv[i] == '--module':
            nsel += 1
            i += 1
            module = int(argv[i])
        elif argv[i] == '--quabo':
            nsel += 1
            i += 1
            quabo_ip_addr = argv[i]
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
    if (nsel > 1):
        print('only one selector allowed')
        usage()

    if quabo_ip_addr:
        quabo_ip_addrs = [quabo_ip_addr]
    else:
        quabo_ip_addrs = config_file.get_quabo_ip_addrs(obs_config, dome, module)
    do_op(quabo_ip_addrs, op)
