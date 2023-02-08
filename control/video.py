#! /usr/bin/env python3

# video.py [--module N] [--ph]
# run (only) while a recording is in progress.
# Shows recent frames from it.
#   --module N      show frames from module N (default: first one)
#   --ph            show pulse height images (default: image mode)
#
# It does this by creating a process on the DAQ node that
# monitors the relevant PFF file and returns frames from the end

import sys, subprocess, pty

import show_pff, util

sys.path.insert(0, '../util')

import config_file, pff

def main(quabo_uids, module_id, dp):
    if module_id < 0:
        dome = quabo_uids['domes'][0]
        module = dome['modules'][0]
        module_id = module['id']
        node = daq_config['daq_nodes'][0]
    else:
        found = False
        for dome in quabo_uids['domes']:
            for module in dome['modules']:
                if module['id'] == module_id:
                    found = True
                    break
            if found: break
        if not found:
            print('no such module %d'%module_id)
            return
    daq_node = module['daq_node']
        
    cmd = 'cd %s; ./video_daq.py --module %d --dp %s'%(
            daq_node['data_dir'], module_id, dp
    )
    print(cmd)
    process = subprocess.Popen(['ssh',
        '%s@%s'%(node['username'], node['ip_addr']),
        cmd,
        ],
        shell=False, stdout = subprocess.PIPE
    )
    ph = False
    if dp == 'img16':
        image_size = 32
        bpp = 2
    elif dp == 'img8':
        image_size = 32
        bpp = 1
    elif dp == 'ph':
        image_size = 16
        ph = True
    while True:
        j = ''
        while True:
            line = process.stdout.readline()
            line = line.decode()
            if line == '\n':
                break
            j += line
        show_pff.print_json(j, ph, False)
        #print('got header')
        img = pff.read_image(process.stdout, image_size, bpp)
        show_pff.image_as_text(img, image_size, bpp, 0, 256)
        if process.poll() is not None:
            break

i = 1
module_id = -1
argv = sys.argv
ph = False
while i<len(argv):
    if argv[i] == '--module':
        i += 1
        module_id = int(argv[i])
    elif argv[i] == '--ph':
        ph = True
    i += 1


daq_config = config_file.get_daq_config()
quabo_uids = config_file.get_quabo_uids()
config_file.associate(daq_config, quabo_uids)
data_config = config_file.get_data_config()
if ph:
    if 'pulse_height' not in data_config.keys():
        raise Exception('no pulse height being recorded')
    dp = 'ph'
else:
    if 'image' not in data_config.keys():
        raise Exception('no image data being recorded')
    bits_pixel = data_config['image']['quabo_sample_size']
    if bits_pixel == 16:
        dp = 'img16'
    else:
        dp = 'img8'

main(quabo_uids, module_id, dp)
