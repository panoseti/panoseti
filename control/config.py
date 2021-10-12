#! /usr/bin/env python

# Configure quabos
# based on matlab/initq.m, startq*.py
# options:
# --show        show list of domes/modules/quabos
# --dome N      select dome
# --module N    select module
# --quabo N     select quabo
# --ping        ping selected quabos
# --reboot      reboot selected quabos
# --loads       load silver firmware in selected quabos
# --loadg       load gold firmware in selected quabos

firmware_silver = 'quabo_0116C_23CBEAFB.bin'
firmware_gold = 'quabo_GOLD_23BD5DA4.bin'

import config_file, sys, os
from panoseti_tftp import tftpw

def show_quabos(obs_config):
    for dome in obs_config['domes']:
        print('dome %s'%dome['num'])
        for module in dome['modules']:
            print('   module %s'%module['num'])
            print('      ID: %s'%module['id'])
            for quabo in module['quabos']:
                print('      quabo %s'%quabo['num'])
                print('         IP addr: %s'%quabo['ip_addr'])

def do_op(quabos, op):
    for quabo in quabos:
        print(op, quabo['ip_addr'])
        x = tftpw(quabo['ip_addr'])
        if op == 'reboot':
            x.reboot()
        elif op == 'loads':
            x.put_bin_file(firmware_silver)
        elif op == 'loadg':
            x.put_bin_file(firmware_gold, 0x0)
        elif op == 'ping':
            ret = os.system('ping -c 1 %s'%quabo['ip_addr'])
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
    quabo = -1
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
        elif argv[i] == '--loadg':
            nops += 1
            op = 'loadg'
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
            quabo = int(argv[i])
        else:
            raise Exception('bad arg %s'%argv[i])
        i += 1

    if (nops == 0):
        raise Exception('no op specified')
    if (nops > 1):
        raise Exception('must specify a single op')
    if (nsel > 1):
        raise Exception('only one selector allowed')

    quabos = config_file.get_quabos(obs_config, dome, module, quabo)
    do_op(quabos, op)
