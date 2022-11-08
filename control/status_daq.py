#! /usr/bin/env python3

# return a JSON description of the recording status on a DAQ node:
# - whether hashpipe is running
# - whether a run in progress
# - free space on disks
#
# runs in the data dir on the DAQ node

import os, json
import util

def status():
    x = {}
    x['hashpipe_running'] = 1 if util.is_hashpipe_running() else 0

    if os.path.exists(util.daq_run_name_filename):
        with open(util.daq_run_name_filename) as f:
            run_name = f.read().strip()
            x['current_run'] = run_name
        if os.path.exists(run_name):
            used = util.disk_usage(run_name)
            x['current_run_disk'] = used

    # for each volume:
    # - name
    # - free space
    # - list of volumes that go there; -1 if default
    vols = {}
    for f in os.listdir('.'):
        y = f.split('_')
        if len(y) != 2: continue
        if y[0] != 'module': continue
        if not y[1].isnumeric(): continue
        modnum = int(y[1])
        n = os.path.realpath(f)
        n = n.split('/')
        n = n[0:3]
        name = '/'.join(n)
        if name in vols.keys():
            vol = vols[name]
            vol['modules'].append(modnum)
        else:
            vol = {}
            vol['modules'] = [modnum]
            f = util.free_space(name)
            vol['free'] = f
            vols[name] = vol
    n = os.path.realpath('.')
    n = n.split('/')
    n = n[0:3]
    name = '/'.join(n)
    if name in vols.keys():
        vols[name]['modules'].append(-1)
    else:
        vol = {}
        vol['modules'] = [-1]
        vol['free'] = util.free_space(name)
        vols[name] = vol

    x['vols'] = vols
    print(json.dumps(x))

status()
