#! /usr/bin/env python3

# return a JSON description of the recording status on a DAQ node:
# - whether hashpipe is running
# - whether a run in progress
# - free space on disks
#
# runs in the data dir on the DAQ node

import json
import os
from typing import Any

import util


def status():
    x: dict[str, Any] = {}
    x['hashpipe_running'] = 1 if util.is_hashpipe_running() else 0

    run_name = util.daq_get_run_name()
    if run_name:
        x['current_run'] = run_name

        # the following is invalid.  need to look in module_N/run
        if os.path.exists(run_name):
            used = util.disk_usage(run_name)
            x['current_run_disk'] = used

    # for each volume:
    # - name
    # - free space
    # - list of modules that go there; -1 if default
    vols: dict[str, Any] = {}
    for f in os.listdir('.'):
        y = f.split('_')
        if len(y) != 2:
            continue
        if y[0] != 'module':
            continue
        if not y[1].isnumeric():
            continue
        modnum = int(y[1])
        module_path = os.path.realpath(f)
        module_parts = module_path.split('/')
        module_parts = module_parts[0:3]
        name = '/'.join(module_parts)
        if name in vols:
            vol = vols[name]
            vol['modules'].append(modnum)
        else:
            vol = {}
            vol['modules'] = [modnum]
            free_size = util.free_space(name)
            vol['free'] = free_size
            vols[name] = vol
    cwd_path = os.path.realpath('.')
    cwd_parts = cwd_path.split('/')
    cwd_parts = cwd_parts[0:3]
    name = '/'.join(cwd_parts)
    if name in vols:
        vol = vols[name]
        vol['modules'].append(-1)
    else:
        vol = {}
        vol['modules'] = [-1]
        free_size = util.free_space(name)
        vol['free'] = free_size
        vols[name] = vol
    x['vols'] = vols
    print(json.dumps(x))

status()
