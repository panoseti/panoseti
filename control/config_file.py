#! /usr/bin/env python3

# functions to read and parse config files

import json

obs_config_filename = 'obs_config.json'
daq_config_filename = 'daq_config.json'
data_config_filename = 'data_config.json'
quabo_uids_filename = 'quabo_uids.json'

# assign sequential numbers to domes and modules
#
def assign_numbers(c):
    ndome = 0
    nquabo = 0
    for dome in c['domes']:
        dome['num'] = ndome
        ndome += 1
        nmodule = 0
        for module in dome['modules']:
            module['num'] = nmodule
            nmodule += 1

# input: a string of the form "0-2, 5-6"
# output: a list of integers, e.g. 0,1,2,5,6
#
def string_to_list(s):
    out = []
    parts = s.split(',')
    for part in parts:
        nums = part.split('-')
        if len(nums) > 1:
            a = int(nums[0])
            b = int(nums[1])
            for i in range(a,b+1):
                out.append(i)
        else:
            out.append(int(nums[0]))
    return out

# in DAQ node objects, expand module range strings
# to list of module numbers
#
def expand_ranges(daq_config):
    for node in daq_config['daq_nodes']:
        node['module_nums'] = string_to_list(node['module_nums'])
            

# given a module number, find the DAQ node that's handling it
#
def module_num_to_daq_node(daq_config, module_num):
    for node in daq_config['daq_nodes']:
        if module_num in node['module_nums']:
            return node
    raise Exception("no DAQ node is handling module number %d"%module_num)

def get_obs_config():
    with open(obs_config_filename) as f:
        s = f.read()
    c = json.loads(s)
    assign_numbers(c)
    return c

def get_daq_config():
    with open(daq_config_filename) as f:
        s = f.read()
    c = json.loads(s)
    expand_ranges(c)
    return c

def get_data_config():
    with open(data_config_filename) as f:
        c = f.read()
    return json.loads(c)

def get_quabo_uids():
    with open(quabo_uids_filename) as f:
        s = f.read()
    c = json.loads(s)
    assign_numbers(c)
    return c

# return list of modules from obs_config
#
def get_modules(c):
    modules = []
    for dome in c['domes']:
        for module in dome['modules']:
            modules.append(module)
    return modules

if __name__ == "__main__":
    c = get_daq_config()
    n = c['daq_nodes'][0]
    print(n)
