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
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../util')
import pff
sys.path.append('../control')
import config_file

import BirdieSource
import ModuleView


def init_birdies():
    pass

def init_module():
    pass

def main():
    pass


if __name__ == 'main':
    main()