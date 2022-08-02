#! /usr/bin/env python

# write images for all data files

import os
import sys
sys.path.append('../util')
import pff

def do_run(run):
    for f in os.listdir('data/%s'%run):
        if pff.is_pff_file(f):
            if pff.pff_file_type(f) != 'img16':
                continue

            # generate images.bin
            #
            cmd = 'write_images < data/%s/%s > derived/%s/%s/images.bin'%(run,f,run,f)
            print(cmd)
            os.system(cmd)

            # generate images.mp4
            #
            nframes = 1000
# see https://stackoverflow.com/questions/20743070/ffmpeg-compressed-mp4-video-not-playing-on-mozilla-firefox-with-a-file-is-corru
            cmd = 'php pipe_images.php derived/%s/%s/images.bin 0 65536 %d | ffmpeg -y -f rawvideo -pix_fmt argb -s 128x128 -r 25 -i - -pix_fmt yuv420p -c:v libx264 -movflags +faststart -vf scale=512:512 derived/%s/%s/images_0_65536_%d.mp4'%(run,f,nframes,run,f, nframes)
            print(cmd)
            os.system(cmd)

def main():
    for run in os.listdir('data'):
        if pff.is_pff_dir(run):
            do_run(run)

main()
