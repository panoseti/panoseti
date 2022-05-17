#! /usr/bin/env python3

# run pulse analysis on a few pixels from all data files

import os
import sys
sys.path.append('../util')
import pff

# make histogram data for pixel values
#
def make_hist(infile, outfile):
    first = True
    counts = {}
    with open(infile) as f:
        for line in f:
            if first:
                first = False
                continue
            x = line.split(',')
            if len(x) < 2:
                break
            v = int(float(x[1]))
            if v == 0:
                continue;
            if v in counts:
                counts[v] += 1
            else:
                counts[v] = 1
    f = open(outfile, "w")
    f.write('value,count\n')
    for val in counts:
        f.write('%d,%d\n'%(val, counts[val]))
    f.close()
            
def do_run(d):
    for f in os.listdir('data/%s'%d):
        if not pff.is_pff_file(f):
            continue
        if pff.pff_file_type(f) != 'img16':
            continue
        print('processing file ', f)
        for pixel in {0, 64, 128, 256, 320, 384}:
            cmd = 'pulse --file data/%s/%s --pixel %d'%(d, f, pixel)
            print(cmd)
            os.system(cmd)
            make_hist(
                'derived/%s/%s/%d/all_0'%(d, f, pixel),
                'derived/%s/%s/%d/value_hist'%(d, f, pixel)
            )

def main():
    for f in os.listdir('data'):
        if pff.is_pff_dir(f):
            print('processing run',f);
            do_run(f);

main()
