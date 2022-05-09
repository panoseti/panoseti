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
            if counts.has_key(v):
                counts[v] += 1
            else:
                counts[v] = 1
    f = open(outfile, "w")
    f.write('value,count\n')
    for val in counts:
        f.write('%d,%d\n'%(val, counts[val]))
    f.close()
            
def do_hdf5(f):
    for module in range(2):
        for pixel in {0, 64, 128, 256, 320, 384}:
            cmd = 'pulse --file PANOSETI_DATA/%s --module %d --pixel %d'%(f, module, pixel)
            print(cmd)
            os.system(cmd)
            make_hist(
                'pulse_out/%s/%d/%d/all_0'%(f, module, pixel),
                'pulse_out/%s/%d/%d/value_hist'%(f, module, pixel)
            )

def do_pff(d):
    for f in os.listdir('PANOSETI_DATA/%s'%d):
        if not f.endswith('.pff'):
            continue
        print('name', f)
        n = pff.parse_name(f)
        if 'dp' not in n.keys():
            continue
        if n['dp'] != '1':
            continue
        for pixel in {0, 64, 128, 256, 320, 384}:
            cmd = 'pulse --file PANOSETI_DATA/%s/%s --pixel %d'%(d, f, pixel)
            print(cmd)
            os.system(cmd)
            make_hist(
                'pulse_out/%s/%s/%d/all_0'%(d, f, pixel),
                'pulse_out/%s/%s/%d/value_hist'%(d, f, pixel)
            )

def main():
    for f in os.listdir('PANOSETI_DATA'):
        if f.endswith('.h5'):
            print('skipping ',f);
#do_hdf5(f)
        elif f.endswith('.pffd'):
            print('doing ',f);
            do_pff(f);

main()
