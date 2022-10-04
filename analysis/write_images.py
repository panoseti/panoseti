#! /usr/bin/env python3

# write_images.py --run X --seconds N
#
# for a run's image files:
# - write images.bin (for frame browser)
# - write video (.mp4)
# do this for the first N seconds of data

import os, sys, getpass
sys.path.append('../util')
import pff
from analysis_util import *

def do_run(run, params, username):
    analysis_dir = make_analysis_dir(ANALYSIS_TYPE_VISUAL, run)
    nframes = img_seconds_to_frames(run, params['seconds'])
    for f in os.listdir('data/%s'%run):
        if not pff.is_pff_file(f): continue
        t = pff.pff_file_type(f)
        if t == 'img16':
            bytes_per_pixel = 2
        elif t == 'img8':
            bytes_per_pixel = 1
        else:
            raise Exception('bad file type %s'%t)

        file_path = 'data/%s/%s'%(run, f)
        if os.path.getsize(file_path) == 0: continue;

        file_attrs = pff.parse_name(f)
        module = file_attrs['module']
        module_dir =  make_dir('%s/module_%s'%(analysis_dir, module))

        # generate images.bin
        #
        cmd = './write_images --bytes_per_pixel --nframes %d< %s> %s/images.bin'%(
            bytes_per_pixel, nframes, file_path, module_dir
        )
        print(cmd)
        os.system(cmd)

        # generate images.mp4
        
        # pixel values for black and white;
        # max = 0 means use full range
        # TODO: use quantiles instead
        minval = 0
        maxval = 0
# see https://stackoverflow.com/questions/20743070/ffmpeg-compressed-mp4-video-not-playing-on-mozilla-firefox-with-a-file-is-corru
        cmd = 'php pipe_images.php %s/images.bin %d %d %d %d | ffmpeg -y -f rawvideo -pix_fmt argb -s 128x128 -r 25 -i - -pix_fmt yuv420p -c:v libx264 -movflags +faststart -vf scale=512:512 %s/images.mp4 2>&1'%(
            module_dir, minval, maxval, nframes, bytes_per_pixel
            module_dir
        )
        print(cmd)
        os.system(cmd)
    write_summary(analysis_dir, params, username)

if __name__ == '__main__':
    params = {
        'seconds': 10
    }
    run = None
    username = None
    argv = sys.argv
    i = 1
    while i<len(argv):
        if argv[i] == '--run':
            i += 1
            run = argv[i]
        elif argv[i] == '--seconds':
            i += 1
            params['seconds'] = float(argv[i])
        elif argv[i] == '--username':
            i += 1
            username = argv[i]
        else:
            raise Exception('bad arg: %s'%argv[i])
        i += 1
    if not run:
        raise Exception('no run specified')
    if not username:
        username = getpass.getuser()

    do_run(run, params, username)
