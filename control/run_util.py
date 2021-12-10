# utility functions for recording runs

import os, sys, subprocess, signal

sys.path.insert(0, '../util')

import pff, config_file

hk_pid_file = '.hk_pid'
    # stores the PID of the housekeeping process

run_name_file = '.run_name'
    # stores the name of the current run

def start_hk_recorder(run_name):
    try:
        process = subprocess.Popen(['record_hk.py', run_name])
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
    except:
        print("HK recorder not running")

def write_run_name(run_name):
    with open(run_name_file, 'w') as f:
        f.write(run_name)

def read_run_name():
    with open(run_name_file) as f:
        return f.read()

def make_run_name(obs, run_type):
    return pff.run_dir_name(obs, run_type);

def remove_run_name():
    if os.path.exists(run_name_file):
        os.unlink(run_name_file)

# for now data dir is just ./data/
def data_dir():
    return 'data'

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
            for i in range(4):
                q = module['quabos'][i];
                print("data from quabo %s (%s) -> DAQ node %s"%(
                    q['uid'], quabo_ip_address(q['ip_addr'], i)
                ))

