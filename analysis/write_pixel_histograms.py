DEPRECATED
#! /usr/bin/env python3

# write_pixel_histograms.py
#
# write pixel histograms for all img files

import os, sys
sys.path.append('../util')
import pff, pixel_histogram

def do_file(infile, outfile, nframes):
    hist = pixel_histogram.get_hist(infile, nframes)
    fout = open(outfile, "w");
    for i in range(256):
        fout.write('%d %d\n'%(i*256, hist[i]))
    print('write histogram for %s'%infile)

def do_run(run):
    for f in os.listdir('data/%s'%run):
        if not pff.is_pff_file(f):
            continue
        if pff.pff_file_type(f) != 'img16':
            continue
        do_file(
            'data/%s/%s'%(run, f),
            '%s/%s/%s/pixel_histogram.dat'%(ANALYSIS_ROOT, run, f),
            100
        )

if __name__ == '__main__':
    for run in os.listdir('data'):
        if not pff.is_pff_dir(run): continue
