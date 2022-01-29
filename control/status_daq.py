#! /usr/bin/env python3

# print the recording status on a DAQ node:
# - whether hashpipe is running
# - whether a run in progress
# - bytes recorded so far, and avg rate
# - free space, and time left at avg rate
#
# runs in the data dir on the DAQ node

import os, psutil, shutil

def disk_usage(dir):
    x = 0
    for f in os.listdir(dir):
        x += os.path.getsize('%s/%s'%(dir, f))
    return x

def free_space():
    total, used, free = shutil.disk_usage('.')
    return free

def status():
    if "hashpipe" in (p.name() for p in psutil.process_iter()):
        print('hashpipe is running')
    else:
        print('hashpipe is not running')

    if os.path.exists('current_run_daq'):
        with open('current_run_daq') as f:
            run_name = f.read().strip()
            print('current run: %s'%run_name)
        if os.path.exists(run_name):
            used = disk_usage(run_name)
            print('disk used: %.2fMB'%(used/1.e6))
        else:
            print("run dir doesn't exist");
    else:
        print('no current run')

    print('disk free: %.2fMB'%(free_space()/1.e6))

    
status()
