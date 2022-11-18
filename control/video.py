#! /usr/bin/env python3

import sys, subprocess, pty

sys.path.insert(0, '../util')

import config_file

def main(daq_config):
    node = daq_config['daq_nodes'][0]
    process = subprocess.Popen(['ssh',
        '%s@%s'%(node['username'], node['ip_addr']),
        'cd %s; ./video_daq.py'%(node['data_dir']),
        ],
        shell=False, stdout = subprocess.PIPE
    )
    while True:
        output = process.stdout.read(1)
        if process.poll() is not None:
            break
        if output:
            print(output.decode())

daq_config = config_file.get_daq_config()
main(daq_config)
