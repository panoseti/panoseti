#! /usr/bin/env python3

import json

pixel_map_dir = '../quabos/'
pixel_map_files = ['quabo_pixelmap_maroc2phys_bga.json', 'quabo_pixelmap_maroc2phys_qfp.json', 
                   'quabo_pixelmap_phys2maroc_bga.json', 'quabo_pixelmap_phys2maroc_qfp.json']

SRC_DIM = 16
DST_DIM = 32

with open(pixel_map_dir + pixel_map_files[0],'rb') as f:
    maroc2phys_bga = json.load(f)


with open(pixel_map_dir + pixel_map_files[1],'rb') as f:
    maroc2phys_qfp = json.load(f)


with open(pixel_map_dir + pixel_map_files[2],'rb') as f:
    phy2maroc_bga = json.load(f)


with open(pixel_map_dir + pixel_map_files[3],'rb') as f:
    phys2maroc_qfp = json.load(f)

# get the data index in the data packets
#
def get_data_index(qi, bver, loc):
    '''
    qi: quabo index -- [0,1,2,3]
    bver: board version -- ['bga', 'qfp']
    loc: pixel location in quabo_config
    '''
    if(bver == 'bga'):
        pixel_map = maroc2phys_bga
    elif(bver == 'qfp'):
        pixel_map = maroc2phys_qfp
    else:
        raise Exception('Please specify the board version: bga or qfp')
    
    phy_loc = pixel_map['pixel_map_maroc2phys'][loc[0]][loc[1]]
    i = phy_loc[1] - 1
    j = phy_loc[0] - 1
    if(qi == 0):
        dx = j
        dy = SRC_DIM - i - 1
    elif(qi == 1):
        dx = SRC_DIM - i - 1
        dy = DST_DIM - j - 1
    elif(qi == 2):
        dx = DST_DIM - j - 1
        dy = SRC_DIM + i
    elif(qi == 3):
        dx = i
        dy = j
    
    return dx * SRC_DIM + dy


# get the pixel loc in quabo_config
#
def get_pixel_loc(qi, bver, index):
    if(bver == 'bga'):
        pixel_map = phy2maroc_bga
    elif(bver == 'qfp'):
        pixel_map = phy2maroc_qfp
    else:
        raise Exception('Please specify the board version: bga or qfp')
    
    dx = int(index/SRC_DIM)
    dy = index%SRC_DIM

    if(qi == 0):
        i = SRC_DIM -dy - 1
        j = dx
    elif(qi == 1):
        i = SRC_DIM - dx - 1
        j = DST_DIM - dy - 1
    elif(qi == 2):
        i = SRC - dy
        j = DST_DIM - dx - 1
    elif(qi == 3):
        i = dx
        j = dy

    loc = pixel_map['pixel_map_phys2maroc'][j][i]
    return loc
