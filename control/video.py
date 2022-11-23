#! /usr/bin/env python3

import sys, subprocess, pty

import show_pff

sys.path.insert(0, '../util')

import config_file, pff

def main(module, daq_config, test):
    node = daq_config['daq_nodes'][0]
    cmd = 'cd %s; ./video_daq.py --module %d --bytes_per_pixel %d'%(
            node['data_dir'], module, 2
    )
    if test:
        cmd += ' --test'
    print(cmd)
    process = subprocess.Popen(['ssh',
        '%s@%s'%(node['username'], node['ip_addr']),
        cmd,
        ],
        shell=False, stdout = subprocess.PIPE
    )
    while True:
        j = ''
        while True:
            line = process.stdout.readline()
            line = line.decode()
            if line == '\n':
                break
            j += line
        show_pff.print_json(j, False, False)
        print('got header')
        img = pff.read_image(process.stdout, 32, 2)
        show_pff.image_as_text(img, 32, 2, 0, 256)
        if process.poll() is not None:
            break

i = 1
module = -1
argv = sys.argv
test = False
while i<len(argv):
    if argv[i] == '--module':
        i += 1
        module = int(argv[i])
    elif argv[i] == '--test':
        test = True
    i += 1

if module<0:
    raise Exception('no module specified')
main(module, config_file.get_daq_config(), test)
