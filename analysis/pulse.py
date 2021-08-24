#! /usr/bin/env python

# run pulse analysis on a few pixels from all data files

import os

# make histogram data for pixel values
#
def make_hist(pfile, module, pixel):
    fname = 'pulse_out/%s/%d/%d/value'%(pfile, module, pixel)
    first = True
    counts = {}
    with open(fname) as f:
        for line in f:
            if first:
                first = False
                continue
            x = line.split(',')
            if len(x) < 2:
                break
            v = int(x[1])
            if v == 0:
                continue;
            if counts.has_key(v):
                counts[v] += 1
            else:
                counts[v] = 1
    fname = 'pulse_out/%s/%d/%d/value_hist'%(pfile, module, pixel)
    f = open(fname, "w")
    f.write('value,count\n')
    for val in counts:
        f.write('%d,%d\n'%(val, counts[val]))
    f.close()
            
def main():
    for f in os.listdir('PANOSETI_DATA'):
        for module in range(2):
            for pixel in {0, 64, 128, 256, 320, 384}:
                cmd = 'pulse --file PANOSETI_DATA/%s --module %d --pixel %d'%(f, module, pixel)
                print(cmd)
                os.system(cmd)
                make_hist(f, module, pixel)

main()
