#! /usr/bin/env python

# run pulse analysis on a few pixels from all data files

import os

def main():
    for f in os.listdir('PANOSETI_DATA'):
        for module in range(2):
            for pixel in {0, 64, 128, 256, 320, 384}:
                cmd = 'pulse --file PANOSETI_DATA/%s --module %d --pixel %d'%(f, module, pixel)
                print(cmd)
                os.system(cmd)

main()
