#! /usr/bin/env python3

# pixel_histogram.py infile outfile nframes
#
# read the first nframes of a PFF file;
# write a histogram file of the form
# 0 N0
# 256 N1
# ... 256 total lines
#
# where N0 is the number of pixels with values from 0 to 255 etc.

import os, sys
sys.path.append('../util')
import pff

def do_file(infile, outfile, nframes):
    fin = open(infile, "rb");
    hist = [0]*256
    for i in range(nframes):
        x = pff.read_json(fin)
        if x is None:
            break
        x = pff.read_image(fin, 32, 2)
        if x is None:
            break
        for j in range(1024):
            v = x[j]>>8
            hist[v] += 1

    fout = open(outfile, "w");
    for i in range(256):
        fout.write('%d %d\n'%(i*256, hist[i]))

def main():
    for run in os.listdir('data'):
        if not pff.is_pff_dir(run): continue
        for f in os.listdir('data/%s'%run):
            if not pff.is_pff_file(f):
                continue
            if pff.pff_file_type(f) != 'img16':
                continue

            do_file(
                'data/%s/%s'%(run, f),
                'derived/%s/%s/pixel_histogram.dat'%(run, f),
                100
            )

main()
