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
import math

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
        dec = np.random.uniform(utils.dec_bounds[0], utils.dec_bounds[1])
        birdie_sources.append(BaseBirdieSource(ra, dec))
    return birdie_sources


def init_module():
    m1 = ModuleView(42, 1656443180, 10.3, 44.2, 234, 77, 77, 77)
    return m1


def init_sky_array(num_ra):
    return utils.get_sky_image_array(num_ra)

def main():
    module = init_module()
    utils.reduce_dec_range(module)
    sky_array = init_sky_array(360*50)
    print(sky_array.shape)
    # Generate synthetic sky data
    birdie_sources = init_birdies(4000)
    for b in birdie_sources:
        b.generate_birdie(sky_array, 10)
    kernel = Gaussian2DKernel(x_stddev=5)
    convolved_sky_array = convolve_fft(sky_array, kernel, allow_huge=True)
    """
    # Simulate image mode data
    module.update_center_ra_dec_coords(1694809000)
    module.simulate_all_pixel_fovs(convolved_sky_array)
    module.plot_simulated_image()

    module.update_center_ra_dec_coords(1694809500)
    module.simulate_all_pixel_fovs(convolved_sky_array)
    module.plot_simulated_image()
    # Wraps around...
    module.update_center_ra_dec_coords(1694810000)
    module.simulate_all_pixel_fovs(convolved_sky_array)
    module.plot_simulated_image()
    """
    #1694810000

    print('start time test')
    import time
    total_time = 0
    #2381
    num_reps = 100
    max_counter = 0
    for x in range(num_reps):
        if x % math.ceil(num_reps / 10) == 1:
            print(f'{100 * round(x / num_reps, 1)}%')
            module.plot_simulated_image()
        s = time.time()
        module.update_center_ra_dec_coords(1664938235 + 24 * x)
        module.simulate_all_pixel_fovs(convolved_sky_array)
        e = time.time()
        total_time += e - s
        max_counter = max(max_counter, max(module.simulated_img_arr))
    avg_time = total_time / num_reps
    print(f'Num sims = {num_reps}, avg sim time = {avg_time} s, total sim time = {total_time} s')
    print(f'max_counter = {max_counter}')
    utils.graph_sky_array(module.sky_band)

print("RUNNING")
main()
print("DONE")
if __name__ == 'main':
    pass