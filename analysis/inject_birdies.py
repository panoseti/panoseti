"""
The program uses models of the modules in an observatory and the
celestial sphere to generate birdies and simulate image mode data for a single image file.

TODO:

    - Code driver code:
        - Setup procedure:
            - (import birdie log file.
            - Initialize Module objects for this simulation.
            - Initialize BirdieSource objects.
            - Create or update birdie log file.
            - Open a file object for the imade file.
        - Main loop
            - Check if we’ve reached EOF in any of the image mode files.
            - Birdie generation
            - Simulate module image mode output.
            - Update image frames (if applicable).

birdie config file format
{

}

BirdieSource object json data:
{
type: [type of birdie object]

}


"""
from astropy.coordinates import SkyCoord

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../util')
import pff
sys.path.append('../control')
import config_file

from BirdieSource import BaseBirdieSource
from ModuleView import ModuleView, PixelView
import birdie_injection_utils as utils


np.random.seed(101)


def init_birdies():
    ra = np.random.uniform(0, 360)
    dec = np.random.uniform(-90, 90)
    return BaseBirdieSource(ra, dec)


def init_module():
    pass


def init_sky_array():
    pass


def main():
    num_ra = 360
    arr = utils.get_sky_image_array(num_ra)
    print(arr)
    b = init_birdies()
    b.generate_birdie(arr, 10)
    print(arr)


print("RUNNING")
main()
if __name__ == 'main':
    pass