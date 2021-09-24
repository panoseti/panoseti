#! /usr/bin/env python

# write images for all data files

import os
import sys
sys.path.append('../util')
import pff

def do_pff(d):
    for f in os.listdir('PANOSETI_DATA/%s'%d):
        if f.endswith('.pff'):
            n = pff.parse_name(f)
            if n['dp'] != '2':
                continue
            cmd = 'write_images --file PANOSETI_DATA/%s/%s'%(d,f)
            print(cmd)
            os.system(cmd)
        else:
            print('unrecognized file: %s'%f)

def main():
    for f in os.listdir('PANOSETI_DATA'):
        if f.endswith('.h5'):
            continue
            cmd = 'write_images_h5 --file PANOSETI_DATA/%s'%(f)
            print(cmd)
            os.system(cmd)
        elif f.endswith('.pffd'):
            do_pff(f)
        else:
            print('unrecognized file: %s'%f)
            continue

main()
