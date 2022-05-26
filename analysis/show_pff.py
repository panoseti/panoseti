#! /usr/bin/env python3

# show_pff.py [--min x] [--max x] filename
# show a PFF file (image or pulse height) as text

import sys, random
sys.path.insert(0, '../util')
import pff

def image_as_text(img, img_size, bytes_per_pixel, min, max):
    scale = ' .,-+=#@'
        # 8 chars w/ increasing density
    print('-'*(img_size*2+2))
    for row in range(img_size):
        s = '|'
        for col in range(img_size):
            x = img[row*img_size+col]
            if max:
                y = (x-min)/(max-min)
                if y<0: y=0
                if y>1: y=1
                i = int(y*8)
                if i>=8: i=7
#print(x, y, min, max, i)
            else:
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
    image_as_text(img, 32, 2, 0, 0)

def show_file(fname, img_size, bytes_per_pixel, min, max):
    with open(fname, 'rb') as f:
        i = 0
        while True:
            j = pff.read_json(f)
            if not j:
                break
            print(j.encode())
            img = pff.read_image(f, img_size, bytes_per_pixel)
            print('frame', i)
            image_as_text(img, img_size, bytes_per_pixel, min, max)
            i += 1

#test()

i = 1
fname = ''
min = 0
max = 0
argv = sys.argv
while i<len(argv):
    if argv[i] == '--min':
        i += 1
        min = int(argv[i])  
    elif argv[i] == '--max':
        i += 1
        max = int(argv[i])  
    else:
        fname = argv[i]
    i += 1

dict = pff.parse_name(fname)
dp = dict['dp']
if dp == 'img16' or dp=='1':
    show_file(fname, 32, 2, min, max)
elif dp == 'ph16' or dp=='3':
    show_file(fname, 16, 2, min, max)
else:
    raise Exception("bad data product %s"%dp)

