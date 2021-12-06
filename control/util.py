# utility functions for control scripts

import os

# return true if can ping IP addr
#
def ping(ip_addr):
    return not os.system('ping -c 1 -w 1 -q %s > /dev/null 2>&1'%ip_addr)

# given module base IP address, return IP addr of quabo i
#
def quabo_ip_addr(base, i):
    x = base.split('.')
    x[3] = str(int(x[3])+i)
    return '.'.join(x)

# see if quabo is alive by seeing if we got its UID
#
def is_quabo_alive(module, quabo_uids, i):
    n = module['num']
    for dome in c['domes']:
        for m in dome['modules']:
            if m['num'] == module['num']:
                q = m['quabos'][i]
                return q['uid'] != ''
    raise Exception("no such module")
