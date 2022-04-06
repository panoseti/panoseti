#! /usr/bin/env python3

# functions to read and parse config files

import os,sys,json
import util

obs_config_filename = 'obs_config.json'
daq_config_filename = 'daq_config.json'
data_config_filename = 'data_config.json'
quabo_uids_filename = 'quabo_uids.json'
quabo_info_filename = '../quabos/quabo_info.json'
detector_info_filename = '../quabos/detector_info.json'
quabo_calib_filename = '../quabos/quabo_calib_%s.json'

config_file_names = [
    obs_config_filename, daq_config_filename, data_config_filename, quabo_uids_filename
]

# assign sequential numbers to domes,
# and IDs to modules
#
def assign_numbers(c):
    ndome = 0
    for dome in c['domes']:
        dome['num'] = ndome
        ndome += 1
        for module in dome['modules']:
            module['id'] = util.ip_addr_to_module_id(module['ip_addr'])

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
        node['module_ids'] = string_to_list(node['module_ids'])

# given a module ID, find the DAQ node that's handling it
#
def module_id_to_daq_node(daq_config, module_id):
    for node in daq_config['daq_nodes']:
        if module_id in node['module_ids']:
            return node
    raise Exception("no DAQ node is handling module %d"%module_id)

def check_config_file(name):
    if not os.path.exists(name):
        print("The config file '%s' doesn't exist."%name)
        print("Create a symbolic link from %s to a specific config file, e.g.:"%name)
        print("   ln -s %s_lick.json %s"%(name.split('.')[0], name))

        sys.exit()

def get_obs_config():
    check_config_file(obs_config_filename)
    with open(obs_config_filename) as f:
        s = f.read()
    c = json.loads(s)
    assign_numbers(c)
    return c

def get_daq_config():
    check_config_file(daq_config_filename)
    with open(daq_config_filename) as f:
        s = f.read()
    c = json.loads(s)
    expand_ranges(c)
    return c

def get_data_config():
    check_config_file(data_config_filename)
    with open(data_config_filename) as f:
        c = f.read()
    return json.loads(c)

def get_quabo_uids():
    if not os.path.exists(quabo_uids_filename):
        print("%s is missing.  Run get_uids.py"%quabo_uids_filename)
        sys.exit()
    with open(quabo_uids_filename) as f:
        s = f.read()
    c = json.loads(s)
    assign_numbers(c)
    return c

# get detector info as an array indexed by serialno
#
def get_detector_info():
    check_config_file(detector_info_filename)
    with open(detector_info_filename) as f:
        s = f.read()
    c = json.loads(s)
    d = {}
    for det in c:
        d[str(det['serialno'])] = float(det['operating_voltage'])
    return d;

# get quabo info as an array indexed by uid
#
def get_quabo_info():
    check_config_file(quabo_info_filename)
    with open(quabo_info_filename) as f:
        s = f.read()
    c = json.loads(s)
    d = {}
    for q in c:
        d[q['uid']] = q
    return d;

# get quabo calibration info
#
def get_quabo_calib(serialno):
    path = quabo_calib_filename%serialno
    with open(path) as f:
        s = f.read()
    return json.loads(s)

# return list of modules from obs_config
#
def get_modules(c):
    modules = []
    for dome in c['domes']:
        for module in dome['modules']:
            modules.append(module)
    return modules

# link modules to DAQ nodes:
# - in the daq_config data structure, add a list "modules"
#   to each daq node object, of the module objects
#   in the quabo_uids data structure;
# - in the quabo_uids data structure, in each module object,
#   add a link "daq_node" to the DAQ node that's handling it.
#
def associate(daq_config, quabo_uids):
    for node in daq_config['daq_nodes']:
        node['modules'] = []
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            daq_node = module_id_to_daq_node(daq_config, module['id'])
            daq_node['modules'].append(module)
            module['daq_node'] = daq_node

# show which module is going to which data recorder
#
def show_daq_assignments(quabo_uids):
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            ip_addr = module['ip_addr']
            daq_node = module['daq_node']
            for i in range(4):
                q = module['quabos'][i];
                print("data from quabo %s (%s) -> DAQ node %s"
                    %(q['uid'], util.quabo_ip_addr(ip_addr, i), daq_node['ip_addr'])
                )

if __name__ == "__main__":
    c = get_detector_info()
    print(c)
