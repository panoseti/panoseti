#! /usr/bin/env python3

# show_pff.py [--quantile x] filename
# show a PFF file (image or pulse height) as text
# --quantile: find the x and 1-x quantiles, and use those as limits
#   default: 0.1
# if no filename specified, use 'img'

import sys, random, json
sys.path.insert(0, '../util')
import pff, image_quantiles

def image_as_text(img, img_size, bytes_per_pixel, min, max):
    scale = ' .,-+=#@'
        # 8 chars w/ increasing density
    print('-'*(img_size*2+2))
    for row in range(img_size):
        s = '|'
        for col in range(img_size):
            x = img[row*img_size+col]
            if max != min:
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
#test()

def print_json(j, is_ph, verbose):
    if verbose:
        print(j)
    else:
        j = json.loads(j)
        if is_ph:
            print('quabo %d: pkt_num %d, pkt_utc %d pkt_nsec %d, tv_sec %d, tv_usec %d'%(
                j['quabo_num'], j['pkt_num'],
                j['pkt_utc'], j['pkt_nsec'],
                j['tv_sec'], j['tv_usec']
            ))
        else:
            for i in range(4):
                q = j['quabo_%d'%i]
                print('quabo %d: pkt_num %d, pkt_utc %d pkt_nsec %d, tv_sec %d, tv_usec %d'%(
                    i, q['pkt_num'], q['pkt_utc'], q['pkt_nsec'],
                    q['tv_sec'], q['tv_usec']
                ))
        
def show_file(fname, img_size, bytes_per_pixel, min, max, is_ph, verbose):
    with open(fname, 'rb') as f:
        i = 0
        while True:
            j = pff.read_json(f)
            if not j:
                print('reached EOF')
                break
            print('frame', i)
            print_json(j.encode(), is_ph, verbose)
            img = pff.read_image(f, img_size, bytes_per_pixel)
            image_as_text(img, img_size, bytes_per_pixel, min, max)
            i += 1
            input('Enter for next frame')

def usage():
    print("usage: show_pff.py [--quantile x] [--verbose] file")

def main():
    i = 1
    fname = None
    quantile = .1
    verbose = False

    argv = sys.argv
    while i<len(argv):
        if argv[i] == '--quantile':
            i += 1
            min = float(argv[i])  
        elif argv[i] == '--verbose':
            verbose = True
        else:
            fname = argv[i]
        i += 1

    if not fname:
        usage()
        return

    if fname=='img':
        dp = 'img16'
    elif fname=='ph':
        dp = 'ph16'
    else:
        dict = pff.parse_name(fname)
        dp = dict['dp']

    if dp == 'img16' or dp=='1':
        image_size = 32
        bytes_per_pixel = 2
        is_ph = False
    elif dp == 'ph16' or dp=='3':
        image_size = 16
        bytes_per_pixel = 2
        is_ph = True
    else:
        raise Exception("bad data product %s"%dp)

    [min, max] = image_quantiles.get_quantiles(
        fname, image_size, bytes_per_pixel, quantile
    )
    print('pixel 10/90 percentiles: %d, %d'%(min, max))
    show_file(fname, image_size, bytes_per_pixel, min, max, is_ph, verbose)

main()
