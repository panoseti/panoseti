#! /usr/bin/env python

# functions to read and parse config files

import json

obs_config_filename = 'obs_config.json'
daq_config_filename = 'daq_config.json'
data_config_filename = 'data_config.json'

# given an IP address, return one with offset i
#
def quabo_ip_addr(base, i):
    x = base.split('.')
    x[3] = str(int(x[3])+i)
    return '.'.join(x)

# assign sequential numbers to domes, modules, and quabos
#
def assign_numbers(c):
    ndome = 0
    nmodule = 0
    nquabo = 0
    for dome in c['domes']:
        dome['num'] = ndome
        ndome += 1
        for module in dome['modules']:
            module['num'] = nmodule
            nmodule += 1

def get_obs_config():
    with open(obs_config_filename) as f:
        s = f.read()
    c = json.loads(s)
    assign_numbers(c)
    return c

def get_daq_config():
    with open(daq_config_filename) as f:
        c = f.read()
    return json.loads(c)

def get_data_config():
    with open(data_config_filename) as f:
        c = f.read()
    return json.loads(c)

# return list of quabo IP addrs from obs_config
# idome, imodule: -1 if not specified
#
def get_quabo_ip_addrs(c, idome, imodule):
    ip_addrs = []
    for dome in c['domes']:
        if idome>=0 and dome['num'] != idome:
            continue
        for module in dome['modules']:
            if imodule>=0 and module['num'] != imodule:
                continue
            ip_addr = module['ip_addr']
            for i in range(4):
                ip_addrs.append(quabo_ip_addr(ip_addr, i))
    return ip_addrs

if __name__ == "__main__":
    c = get_obs_config()
    print(c)
