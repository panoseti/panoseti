#! /usr/bin/env python3

# run image pulse analysis on PFF image file in data/
#
# cmdline args:
#
# --run R               process run R
# --pixels a,b,c        process pixels a,b,c
# --all_pixels          process all pixels
#                       (the above not mutually exclusive)
# --log_all             write complete info (big)
# --seconds X           analyze X seconds of data
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

import os, sys, getpass
sys.path.append('../util')
import pff, config_file
from analysis_util import *

def sort_output_files(dir, nlevels):
    for i in range(nlevels):
        path = '%s/thresh_%d'%(dir, i)
        if os.path.exists(path):
            cmd = 'sort -k 5 -r %s > %s.sorted'%(path, path)
            os.system(cmd)

def do_file(run, analysis_dir, f, params, bits_per_pixel):
    print('processing file ', f)
    file_attrs = pff.parse_name(f)
    module = file_attrs['module']
    module_dir = make_dir('%s/module_%s'%(analysis_dir, module))
    p = '--nlevels %d --win_size %d --thresh %f --bits_per_pixel %d'%(
        params['nlevels'], params['win_size'], params['thresh'], bits_per_pixel
    )
    if params['seconds'] > 0:
        nframes = img_seconds_to_frames(run, params['seconds'])
        p += ' --nframes %d'%nframes
    if params['log_all']:
        p += ' --log_all'
    if params['pixels']:
        for pixel in params['pixels']:
            pixel_dir = make_dir('%s/pixel_%d'%(module_dir, pixel))
            cmd = './img_pulse --infile data/%s/%s --pixel %d --out_dir %s %s'%(
                run, f, pixel, pixel_dir, p
            )
            print(cmd)
            os.system(cmd)
            sort_output_files(pixel_dir, params['nlevels'])
    if params['all_pixels']:
        all_dir = make_dir('%s/all_pixels'%(module_dir))
        cmd = './img_pulse --infile data/%s/%s --out_dir %s %s'%(
            run, f, all_dir, p
        )
        print(cmd)
        os.system(cmd)
        sort_output_files(all_dir, params['nlevels'])

def do_run(run, params, username):
    analysis_dir = make_analysis_dir(ANALYSIS_TYPE_IMAGE_PULSE, run)
    print('processing run', run);
    run_dir = 'data/%s'%run
    for f in os.listdir(run_dir):
        if not pff.is_pff_file(f):
            continue
        if os.path.getsize('%s/%s'%(run_dir, f)) == 0:
            continue;
        if pff.pff_file_type(f) == 'img16':
            do_file(run, analysis_dir, f, params, 16)
        elif pff.pff_file_type(f) == 'img8':
            do_file(run, analysis_dir, f, params, 8)
    write_summary(analysis_dir, params, username)

if __name__ == '__main__':

    # default params
    params = {
        'nlevels': 16,
        'win_size': 64,
        'thresh': 3,
        'pixels': '',
        'seconds': -1,
        'all_pixels': False,
        'log_all': False
    }

    run = None
    username = None
    argv = sys.argv
    i = 1
    while i<len(argv):
        if argv[i] == '--run':
            i += 1
            run = argv[i]
        elif argv[i] == '--all_runs':
            run = 'all'
        elif argv[i] == '--pixels':
            i += 1
            p = argv[i]
            p = p.split(',')
            params['pixels'] = [int(x) for x in p]
        elif argv[i] == '--nlevels':
            i += 1
            params['nlevels'] = int(argv[i])
        elif argv[i] == '--win_size':
            i += 1
            params['win_size'] = int(argv[i])
        elif argv[i] == '--thresh':
            i += 1
            params['thresh'] = float(argv[i])
        elif argv[i] == '--username':
            i += 1
            username = argv[i]
        elif argv[i] == '--all_pixels':
            params['all_pixels'] = True
        elif argv[i] == '--log_all':
            params['log_all'] = True
        elif argv[i] == '--seconds':
            i += 1
            params['seconds'] = float(argv[i])
        else:
            raise Exception('bad arg: %s'%argv[i])
        i += 1
    if not params['all_pixels'] and not params['pixels']:
        raise Exception("no pixels specified")
    if not username:
        username = getpass.getuser()
    if run == 'all':
        for run in os.listdir('data'):
            if pff.is_pff_dir(run):
                do_run(run, params, username);
    elif run:
        do_run(run, params, username)
    else:
        raise Exception('no run specified')
