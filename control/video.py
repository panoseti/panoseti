#! /usr/bin/env python3

import sys, subprocess, pty

sys.path.insert(0, '../util')

import config_file

def main(module, daq_config):
    node = daq_config['daq_nodes'][0]
    cmd = 'cd %s; ./video_daq.py --module %d --bytes_per_pixel %d'%(
            node['data_dir'], module, 2
    )
    print(cmd)
    process = subprocess.Popen(['ssh',
        '%s@%s'%(node['username'], node['ip_addr']),
        cmd,
        ],
        shell=False, stdout = subprocess.PIPE
    )
    while True:
        output = process.stdout.read(1)
        if process.poll() is not None:
            break
        if output:
            print(output.decode())

i = 1
module = -1
argv = sys.argv
while i<len(argv):
    if argv[i] == '--module':
        i += 1
        module = int(argv[i])
if module<0:
    raise Exception('no module specified')
main(module, config_file.get_daq_config())
