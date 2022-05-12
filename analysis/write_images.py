#! /usr/bin/env python

# write images for all data files

import os
import sys
sys.path.append('../util')
import pff

def do_pff(d):
    for f in os.listdir('PANOSETI_DATA/%s'%d):
        if f.endswith('.pff'):
            n = pff.parse_name(f)
            if 'dp' not in n.keys():
                continue
            if n['dp'] != '1':
                continue

            # generate images.bin
            #
            cmd = 'write_images < PANOSETI_DATA/%s/%s > pulse_out/%s/%s/images.bin'%(d,f,d,f)
            print(cmd)
            os.system(cmd)

            # generate images.mp4
            #
            nframes = 1000
            cmd = 'php pipe_images.php pulse_out/%s/%s/images.bin %d | ffmpeg -y -f rawvideo -pix_fmt argb -s 128x128 -r 25 -i - -crf 0 -vf scale=512:512 pulse_out/%s/%s/images.mp4'%(d,f,nframes,d,f)
            print(cmd)
            os.system(cmd)
        else:
            print('unrecognized file: %s'%f)

def main():
    for f in os.listdir('PANOSETI_DATA'):
        if f.endswith('.h5'):
            continue
            cmd = 'write_images_h5 --file PANOSETI_DATA/%s'%(f)
            print(cmd)
            os.system(cmd)
        elif f.endswith('.pffd'):
            do_pff(f)
        else:
            print('unrecognized file: %s'%f)
            continue

main()
