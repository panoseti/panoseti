#! /usr/bin/env python3

# show the status of a recording run

import os
from datetime import UTC, datetime
from typing import Any

from utils import config_file, util


# ---------- logging setup ----------
def ut_now_str() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

def ut_date_str() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")

def log_print(*args: Any, **kwargs: Any) -> None:
    msg = " ".join(str(a) for a in args)
    line = f"[{ut_now_str()}] {msg}"

    # console
    print(line, **kwargs)

    # file
    yyyymmdd = ut_date_str()
    log_dir = f"/mnt/data11/data/palomar/L0/{yyyymmdd}/obslogs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"datarec_{yyyymmdd}.log")
    with open(log_path, "a") as f:
        f.write(line + "\n")

# ---------- main logic ----------
def status() -> None:
    run_name = util.read_run_name()
    if run_name:
        log_print(f'Run in progress: {run_name}')
    else:
        log_print("No run is in progress")

    if util.is_hk_recorder_running():
        log_print('HK recorder is running')
    else:
        log_print('HK recorder is not running')

    # in theory should use config files in run dir
    config_file.get_obs_config()
    daq_config = config_file.get_daq_config()
    quabo_uids = config_file.get_quabo_uids()
    config_file.get_data_config()
    config_file.associate(daq_config, quabo_uids)
    util.local_ip()

    for node in daq_config['daq_nodes']:
        if not node['modules']:
            continue
        ip_addr = node['ip_addr']
        log_print(f'status on DAQ node {ip_addr}:')
        j = util.get_daq_node_status(node)

        if j['hashpipe_running']:
            log_print('   hashpipe is running')
        else:
            log_print('   hashpipe is not running')

        if 'current_run' in j.keys():
            log_print('   current run:', j['current_run'])
            if 'current_run_disk' in j.keys():
                log_print('   disk usage:', j['current_run_disk'])
            else:
                log_print("   run dir doesn't exist")
        else:
            log_print('   no current run')

        vols = j['vols']
        log_print('   volumes:')
        for name in vols.keys():
            vol = vols[name]
            log_print('      name:', name)
            log_print('         free space: %.2fGB' % (vol['free'] / 1e9))
            log_print('         modules:', vol['modules'])

status()

