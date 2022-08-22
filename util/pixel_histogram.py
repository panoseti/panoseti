# read the first N frames of a PFF file.
# compute the x and 1-x quantiles of the pixels

import os, sys
import pff

def get_values(file, image_size, bytes_per_pixel, nframes=100):
    fin = open(file, "rb");
    values = []
    for i in range(nframes):
        x = pff.read_json(fin)
        if x is None:
            break
        x = pff.read_image(fin, image_size, bytes_per_pixel)
        if x is None:
            break
        for j in range(image_size*image_size):
            values.append(x[j])
    fin.close()
    return values

def get_quantiles(file, img_size, bytes_per_pixel, x):
    values = get_values(file, img_size, bytes_per_pixel)
    n = len(values)
    values.sort()
    return [values[int(n*x)], values[int(n*(1-x))] ]
