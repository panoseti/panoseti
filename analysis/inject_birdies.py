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

TODO: Restrict sky_array to just the strip used by the module.

"""
from astropy.coordinates import SkyCoord

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from BirdieSource import BaseBirdieSource
from ModuleView import ModuleView
import birdie_injection_utils as utils
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

sys.path.append('../util')
import pff
sys.path.append('../control')
import config_file

np.random.seed(383)


def init_birdies(num):
    birdie_sources = []
    for x in range(num):
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        birdie_sources.append(BaseBirdieSource(ra, dec))
    return birdie_sources


def init_module():
    m1 = ModuleView(42, 1656443180, 10.3, 44.2, 234, 77, 77, 77)
    return m1


def init_sky_array(num_ra):
    return utils.get_sky_image_array(num_ra)


def main():
    sky_array = init_sky_array(3600)
    module = init_module()
    # Generate synthetic sky data
    birdie_sources = init_birdies(10000)
    for b in birdie_sources:
        b.generate_birdie(sky_array, 10)
    kernel = Gaussian2DKernel(x_stddev=1)
    convolved_sky_array = convolve_fft(sky_array, kernel)

    # Simulate image mode data
    module.update_center_ra_dec_coords(1694809000)
    module.simulate_all_pixel_fovs(convolved_sky_array)
    module.plot_32x32_image()

    module.update_center_ra_dec_coords(1694809500)
    module.simulate_all_pixel_fovs(convolved_sky_array)
    module.plot_32x32_image()

    # Wraps around...
    module.update_center_ra_dec_coords(1694810000)
    module.simulate_all_pixel_fovs(convolved_sky_array)
    module.plot_32x32_image()

    print('start time test')
    import time
    s = time.time()
    for x in range(1000):
        module.update_center_ra_dec_coords(1694810000 + 60*x)
        module.simulate_all_pixel_fovs(convolved_sky_array)
    print(f'Average per simulation {(time.time() - s) / 100}')

    module.simulate_all_pixel_fovs(sky_array, plot_fov=True)

print("RUNNING")
main()
print("DONE")
if __name__ == 'main':
    pass