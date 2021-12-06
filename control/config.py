#! /usr/bin/env python

# Configure and reboot quabos
# options:
# --show        show summary of config file (list of domes and modules)
# --dome N      select dome
# --module N    select module
# --quabo N     select quabo (N=0..3)
# --ping        ping selected quabos
# --reboot      reboot selected quabos
# --get_uids    get UIDs of quabos, write to quabo_uids.json
# --loads       load silver firmware in selected quabos
# --loadg       load gold firmware in selected quabos
#               DEPRECATED - dangerous
# based on matlab/initq.m, startq*.py

firmware_silver = 'quabo_0116C_23CBEAFB.bin'
firmware_gold = 'quabo_GOLD_23BD5DA4.bin'

import config_file, sys, os, quabo_driver, struct
from panoseti_tftp import tftpw

def usage():
    print('''usage:
--show        show list of domes/modules/quabos
--dome N      select dome
--module N    select module
--quabo N     select quabo (N=0..3)
--ping        ping selected quabos
--reboot      reboot selected quabos
--get_uids    get UIDs of quabos, write to quabo_uids.json
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
                quabo_ip = config_file.quabo_ip_addr(ip_addr, i)
                print('      quabo %d'%quabo_num)
                print('         IP addr: %s'%quabo_ip)

# return true if can ping IP addr
#
def ping(ip_addr):
    return not os.system('ping -c 1 -w 1 -q %s > /dev/null 2>&1'%ip_addr)

def do_reboot(modules, quabo_num):
    # need to reboot quabos in order 0..3
    # could do this in parallel across modules; for now, do it serially
    #
    for module in modules:
        for i in range(4):
            if quabo_num>=0 and i != quabo_num:
                continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            if not ping(ip_addr):
                print("can't ping %s"%ip_addr)
                continue
            x = tftpw(ip_addr)
            x.reboot()

            # wait for a housekeeping packet
            #
            quabo = quabo_driver.QUABO(ip_addr)
            while True:
                if quabo.read_hk_packet():
                    break
            quabo.close()

def do_loads(modules, quabo_num):
        for module in modules:
            for i in range(4):
                if quabo_num>=0 and i != quabo_num:
                    continue
                ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
                if not ping(ip_addr):
                    print("can't ping %s"%ip_addr)
                    continue
                x = tftpw(ip_addr)
                x.put_bin_file(firmware_silver)

def do_loadg(modules,quabo_num):
    print("not supported");
    #x.put_bin_file(firmware_gold, 0x0)

def do_ping(modules,quabo_num):
    for module in modules:
        for i in range(4):
            if quabo_num>=0 and i != quabo_num:
                continue
            ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
            if ping(ip_addr):
                print("pinged %s"%ip_addr)
            else:
                print("can't ping %s"%ip_addr)

# return quabo UID as hex string
#
def get_uid(ip_addr):
    print("get uid", ip_addr)
    x = tftpw(ip_addr)
    x.get_flashuid()
    with open('flashuid', 'rb') as f:
        i = struct.unpack('q', f.read(8))
        return "%x"%(i[0])

def get_uids(obs_config):
    f = open('quabo_uids.json', 'w')
    f.write(
'''
[
    "domes": [
''')
    dfirst = True
    for dome in obs_config['domes']:
        if not dfirst:
            f.write(',')
        dfirst = False
        f.write(
'''
        {
            "modules": [
''')
        mfirst = True
        for module in dome['modules']:
            if not mfirst:
                f.write(',')
            mfirst = False
            f.write(
'''
                {
                    "quabos": [
''')
            for i in range(4):
                ip_addr = config_file.quabo_ip_addr(module['ip_addr'], i)
                if not ping(ip_addr):
                    uid = ''
                else:
                    uid = get_uid(ip_addr)
                f.write(
'''
                        {
                            "uid": "%s"
                        }%s
'''%(uid, ('' if i==3 else ',')))
            f.write(
'''
                    ]
                }
''')
        f.write(
'''
            ]
        }
''')
    f.write(
'''
    ]
}
''')
    f.close()

if __name__ == "__main__":
    argv = sys.argv
    nops = 0
    nsel = 0
    dome = -1
    module = -1
    quabo_num = -1
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
        elif argv[i] == '--get_uids':
            nops += 1
            op = 'get_uids'
        elif argv[i] == '--dome':
            nsel += 1
            i += 1
            dome = int(argv[i])
        elif argv[i] == '--module':
            nsel += 1
            i += 1
            module = int(argv[i])
        elif argv[i] == '--quabo':
            i += 1
            quabo_num = int(argv[i])
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

    modules = config_file.get_modules(obs_config, dome, module)
    if op == 'get_uids':
        get_uids(obs_config)
    elif op == 'reboot':
        do_reboot(modules, quabo_num)
    elif op == 'loads':
        do_loads(modules, quabo_num)
    elif op == 'ping':
        do_ping(modules, quabo_num)
