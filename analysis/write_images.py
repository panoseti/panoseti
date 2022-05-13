#! /usr/bin/env python

# write images for all data files

import os
import sys
sys.path.append('../util')
import pff

def do_pff(d):
    for f in os.listdir('PANOSETI_DATA/%s'%d):
        if pff.is_pff_file(f):
            n = pff.parse_name(f)
            if 'dp' not in n.keys():
                continue
            if n['dp'] != 'img16':
                continue

            # generate images.bin
            #
            cmd = 'write_images < PANOSETI_DATA/%s/%s > pulse_out/%s/%s/images.bin'%(d,f,d,f)
            print(cmd)
            os.system(cmd)

            # generate images.mp4
            #
            nframes = 1000
# see https://stackoverflow.com/questions/20743070/ffmpeg-compressed-mp4-video-not-playing-on-mozilla-firefox-with-a-file-is-corru
            cmd = 'php pipe_images.php pulse_out/%s/%s/images.bin %d | ffmpeg -y -f rawvideo -pix_fmt argb -s 128x128 -r 25 -i - -pix_fmt yuv420p -c:v libx264 -movflags +faststart -vf scale=512:512 pulse_out/%s/%s/images.mp4'%(d,f,nframes,d,f)
            print(cmd)
            os.system(cmd)

def main():
    for f in os.listdir('PANOSETI_DATA'):
        if pff.is_pff_dir(f):
            do_pff(f)

main()
