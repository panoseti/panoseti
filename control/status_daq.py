#! /usr/bin/env python3

# print the recording status on a DAQ node:
# - whether hashpipe is running
# - whether a run in progress
# - bytes recorded so far, and avg rate
# - free space, and time left at avg rate
#
# runs in the data dir on the DAQ node

import os
import util

def status():
    if util.is_hashpipe_running():
        print('hashpipe is running')
    else:
        print('hashpipe is not running')

    if os.path.exists(util.daq_run_name_filename):
        with open(util.daq_run_name_filename) as f:
            run_name = f.read().strip()
            print('current run: %s'%run_name)
        if os.path.exists(run_name):
            used = util.disk_usage(run_name)
            print('disk used: %.2fMB'%(used/1.e6))
        else:
            print("run dir doesn't exist");
    else:
        print('no current run')

    print('disk free:')
    vols = {}
    for f in os.listdir('.'):
        if 'module_' in f:
            x = os.path.realpath(f)
            x = x.split('/')
            x = x[0:3]
            v = '/'.join(x)
            print('   %s: volume %s'%(f, v))
            if v not in vols.keys():
                f = util.free_space(v)
                vols[v] = f
    for v, f in vols.items():
        print('   volume %s: %.0f GB free'%(v, f/1e9))
    
status()
