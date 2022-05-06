#! /usr/bin/env python3

# show_image_file.py filename
# show a PFF image file as text

import sys, random
sys.path.insert(0, '../util')
import pff

def image_16_as_text(img):
    scale = ' .,-+=#@'
        # 8 chars w/ increasing density
    print('------------------------------------------------------------------')
    for row in range(32):
        s = '|'
        for col in range(32):
            x = img[col+32*row]
            i = int(x/8192)
            if x>0:
                i=1
            s += scale[i]
            s += ' '
        s += '|'
        print(s)
    print('------------------------------------------------------------------')


def test():
    img = [0]*1024
    for i in range(1024):
        img[i] = random.randrange(2**16)
    image_16_as_text(img)

def show_file(fname):
    with open(fname, 'rb') as f:
        i = 0
        while True:
            if not pff.read_json(f):
                break
            img = pff.read_image_16(f)
            print('frame', i)
            image_16_as_text(img)
            i += 1

show_file(sys.argv[1])
#test()
