# control script utilities

import os, sys, subprocess, signal

import config_file
sys.path.insert(0, '../util')

import pff

#-------------- BINARY DATA ---------------

def print_binary(data):
    n = len(data)
    print('got %d bytes'%n)
    for i in range(n):
        print("%d: %d"%(i, data[i]))

#-------------- FILE NAMES ---------------

hk_pid_file = '.hk_pid'
    # stores the PID of the housekeeping process

run_name_file = '.run_name'
    # stores the name of the current run

hk_file_name = 'hk.pff'
    # housekeeping file in run dir

# for now data dir is just ./data/
def get_data_dir():
    return 'data'

#-------------- QUABO OPS ---------------

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
    for dome in quabo_uids['domes']:
        for m in dome['modules']:
            if m['num'] == module['num']:
                q = m['quabos'][i]
                return q['uid'] != ''
    raise Exception("no such module")

#-------------- RECORDING ---------------

def start_hk_recorder(run_name):
    path = '%s/%s/%s'%(data_dir(), run_name, hk_file_name)
    try:
        process = subprocess.Popen(['record_hk.py', path])
    except:
        print("can't launch HK recorder")
        raise

    with open(hk_pid_file, 'w') as f:
        f.write(str(process.pid))

def stop_hk_recorder():
    if not os.path.exists(hk_pid_file):
        return
    with open(hk_pid_file) as f:
        pid = int(f.read())
    try:
        os.kill(pid, signal.SIGKILL)
        print('Stopped HK data recorder')
    except:
        print('HK recorder not running')
    os.unlink(hk_pid_file)

def write_run_name(run_name):
    with open(run_name_file, 'w') as f:
        f.write(run_name)

def read_run_name():
    with open(run_name_file) as f:
        return f.read()

def make_run_name(obs, run_type):
    return pff.run_dir_name(obs, run_type)

def remove_run_name():
    if os.path.exists(run_name_file):
        os.unlink(run_name_file)

# link modules to DAQ nodes:
# - in the daq_config data structure, add a list "modules"
#   to each daq node object, of the module objects
#   in the quabo_uids data structure;
# - in the quabo_uids data structure, in each module object,
#   add a link "daq_node" to the DAQ node that's handling it.
#
def associate(daq_config, quabo_uids):
    for n in daq_config['daq_nodes']:
        n['modules'] = []
    for dome in quabo_uids['domes']:
        for module in dome['modules']:
            daq_node = config_file.module_num_to_daq_node(daq_config, module['num'])
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
                    %(q['uid'], quabo_ip_addr(ip_addr, i), daq_node['ip_addr'])
                )
