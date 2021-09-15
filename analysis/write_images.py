#! /usr/bin/env python

# write images for all data files

import os

def main():
    for f in os.listdir('PANOSETI_DATA'):
        if f.endswith('.h5'):
            cmd = 'write_images_h5 --file PANOSETI_DATA/%s'%(f)
        elif f.endswith('.pff'):
            cmd = 'write_images --file PANOSETI_DATA/%s'%(f)
        else:
            print('unrecognized file: %s'%f)
            continue
        print(cmd)
        os.system(cmd)

main()
