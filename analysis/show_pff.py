#! /usr/bin/env python3

# show_pff.py filename
# show a PFF file (image or pulse height) as text

import sys, random
sys.path.insert(0, '../util')
import pff

def image_as_text(img, img_size, bytes_per_pixel):
    scale = ' .,-+=#@'
        # 8 chars w/ increasing density
    print('-'*(img_size*2+2))
    for row in range(img_size):
        s = '|'
        for col in range(img_size):
            x = img[row*img_size+col]
            if bytes_per_pixel == 2:
                i = x//8192
            else:
                i = x//32
            if x>0 and i==0:
                i=1
            s += scale[i]
            s += ' '
        s += '|'
        print(s)
    print('-'*(img_size*2+2))


def test():
    img = [0]*1024
    for i in range(1024):
        img[i] = random.randrange(2**16)
    image_as_text(img, 32, 2)

def show_file(fname, img_size, bytes_per_pixel):
    with open(fname, 'rb') as f:
        i = 0
        while True:
            j = pff.read_json(f)
            if not j:
                break
            print(j.encode())
            img = pff.read_image(f, img_size, bytes_per_pixel)
            print('frame', i)
            image_as_text(img, img_size, bytes_per_pixel)
            i += 1

#test()

fname = sys.argv[1]
dict = pff.parse_name(fname)
dp = dict['dp']
if dp == 'img16':
    show_file(fname, 32, 2)
elif dp == 'ph16':
    show_file(fname, 16, 2)
else:
    raise Exception("bad data product %s"%dp)

