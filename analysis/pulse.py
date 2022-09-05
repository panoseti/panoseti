#! /usr/bin/env python3

# run image pulse analysis on PFF image files in data/
#
# cmdline args:
#
# --obs_run R           process run R
# --pixels a,b,c        process pixels a,b,c (default all)
# --log_all             write complete info (big)
#   pulse-finding params (see pulse.cpp):
# --nlevels             default 16
# --win_size            default 64
# --thresh              default 3
#
# input: image-mode PFF files in data/R/
# output: put files in analysis/R/img_pulse/P/ or analysis/R/img_pulse/all
#   files (see pulse.cpp for format):
#       params.json: params in JSON format
#       thresh_i: pulses above threshold (i=pulse duration level; 0,1,...)
#       all_i: all pulses and stats

import os
import sys
sys.path.append('../util')
import pff
from analysis_util import *

# default params
params = {
    'ndurs': 16,
    'win_size': 64,
    'thresh': 3
}

def do_file(run, f, pixels):
    print('processing file ', f)
    analysis_dir = make_analysis_dir(ANALYSIS_TYPE_IMAGE_PULSE, run)
    p = '--ndurs %d --win_size %d --thresh %f'%(
        params['ndurs'], params['win_size'], params['thresh']
    )
    if pixels:
        for pixel in pixels:
            pixel_dir = make_dir('%s/%d'%(analysis_dir, pixel))
            cmd = './pulse --infile data/%s/%s --pixel %d --out_dir %s %s'%(
                run, f, pixel, pixel_dir, p
            )
            print(cmd)
            os.system(cmd)
    else:
        cmd = './pulse --infile data/%s/%s --out_dir %s %s'%(
            run, f, analysis_dir, p
        )
        print(cmd)
        os.system(cmd)
    write_summary(analysis_dir, params)

def do_run(run, pixels):
    print('processing run', run);
    for f in os.listdir('data/%s'%run):
        if not pff.is_pff_file(f):
            continue
        if pff.pff_file_type(f) != 'img16':
            continue
        do_file(run, f, pixels)
        break

if __name__ == '__main__':

    obs_run = None
    pixels = None
    argv = sys.argv
    i = 1
    while i<len(argv):
        if argv[i] == '--obs_run':
            i += 1
            obs_run = argv[i]
        elif argv[i] == '--all_runs':
            obs_run = 'all'
        elif argv[i] == '--pixels':
            i += 1
            p = argv[i]
            p = p.split(',')
            pixels = [int(x) for x in p]
        elif argv[i] == '--ndurs':
            i += 1
            params['ndurs'] = int(argv[i])
        elif argv[i] == '--win_size':
            i += 1
            params['win_size'] = int(argv[i])
        elif argv[i] == '--thresh':
            i += 1
            params['thresh'] = float(argv[i])
        else:
            raise Exception('bad arg: %s'%argv[i])
        i += 1
    print(obs_run)
    if obs_run == 'all':
        for run in os.listdir('data'):
            if pff.is_pff_dir(run):
                do_run(run, pixels);
    elif obs_run:
        do_run(obs_run, pixels)
    else:
        raise Exception('no run specified')
