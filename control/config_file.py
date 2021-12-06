#! /usr/bin/env python

# functions to read and parse config files

import json

obs_config_filename = 'obs_config.json'
daq_config_filename = 'daq_config.json'
data_config_filename = 'data_config.json'

# given module base IP address, return IP addr of quabo i
#
def quabo_ip_addr(base, i):
    x = base.split('.')
    x[3] = str(int(x[3])+i)
    return '.'.join(x)

# assign sequential numbers to domes and modules
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

# return list of modules from obs_config
# idome, imodule: -1 if not specified
#
def get_modules(c, idome, imodule):
    modules = []
    for dome in c['domes']:
        if idome>=0 and dome['num'] != idome:
            continue
        for module in dome['modules']:
            if imodule>=0 and module['num'] != imodule:
                continue
            modules.append(module)
    return modules

if __name__ == "__main__":
    c = get_obs_config()
    print(c)
