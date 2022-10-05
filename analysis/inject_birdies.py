"""
The program uses models of the modules in an observatory and the
celestial sphere to generate birdies and simulate image mode data for a single image file.

TODO:
    - File IO:
        - Add utility methods to import image mode files, read their metadata and image arrays, and write RAW + birdie frames.
        - Most important metadata:
            - Module ID, module orientation (alt-az + observatory GPS), integration time, start time, and end time.
    - Setup procedure:
        - Create or update birdie log file.
        - Open a file object for the imade file.
    - Main loop
        - Check if we’ve reached EOF in any of the image mode files.
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
import math
import time
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from BirdieSource import BaseBirdieSource
from ModuleView import ModuleView
import birdie_injection_utils as utils

sys.path.append('../util')
import pff
sys.path.append('../control')
import config_file

#np.random.seed(383)


def get_birdie_config_vector(param_ranges):
    """Generates a tuple of BirdieSource initialization parameters with uniform distribution
    on the ranges of possible values, provided by param_ranges."""
    unif = np.random.uniform
    # Param order: 'ra', 'dec', 'start_utc', 'end_utc', 'duty_cycle', 'period', 'intensity'
    times = unif(*param_ranges['file_time_range'], 2)
    config_vector = (
        unif(*param_ranges['ra']),
        unif(*param_ranges['dec']),
        min(times),
        max(times),
        unif(*param_ranges['duty_cycle']),
        unif(*param_ranges['period']),
        unif(*param_ranges['intensity'])
    )
    return config_vector


def init_module(start_utc):
    m1 = ModuleView('test', start_utc, 10.3, 44.2, 234, 77, 77, 77)
    return m1


def init_sky_array(num_ra):
    return utils.get_sky_image_array(num_ra)


def init_birdies(num, param_ranges):
    birdie_sources = []
    for x in range(num):
        config_vector = get_birdie_config_vector(param_ranges)
        b = BaseBirdieSource(*config_vector)
        birdie_sources.append(b)
    return birdie_sources


def update_birdies(frame_utc, sky_array, birdie_sources):
    for b in birdie_sources:
        b.generate_birdie(frame_utc, sky_array)


def apply_psf(sky_array, sigma):
    """Apply a 2d gaussian filter to simulate optical distortion."""
    return gaussian_filter(sky_array, sigma=sigma)


def init_birdie_param_ranges(start_utc, end_utc):
    """Param_ranges specifies the range of possible values for each BirdieSource parameter."""
    param_ranges = {
        'ra': (180, 360),
        'dec': utils.dec_bounds,
        'file_time_range': (start_utc, end_utc),
        'duty_cycle': (0.25, 1),#(1e-3, 1e-1),
        'period': (1, 4),
        'intensity': (100, 150),
    }
    return param_ranges


def do_simulation(start_utc, end_utc, module, sky_array, birdie_sources, num_steps):
    max_counter = 0
    step = (end_utc - start_utc) // num_steps
    t = start_utc
    print('Start simulation')
    while t < end_utc:
        utils.show_progress(t, start_utc, end_utc, step, module)
        # Update sky
        sky_array.fill(0)
        update_birdies(t, sky_array, birdie_sources)
        # Sigma = 2 is currently an arbitrary value.
        blurred_sky_array = apply_psf(sky_array, sigma=2)

        # Simulate image mode data
        module.update_center_ra_dec_coords(t)
        module.simulate_all_pixel_fovs(blurred_sky_array, draw_sky_band=True)

        max_counter = max(max_counter, max(module.simulated_img_arr))
        t += step
    print()
    #print(f'\nNum sims = {num_reps}, avg sim time = {avg_time}s, total sim time = {total_time}s')
    #print(f'max image counter = {max_counter}')


def main():
    start_utc = 1656443180
    end_utc = start_utc + 2 * 3600

    # Init ModuleView object
    module = init_module(start_utc)

    # Generate sky array
    utils.reduce_dec_range(module)
    # 30 pixels per degree
    array_resolution = 360 * 30
    sky_array = init_sky_array(array_resolution)

    # Init birdies and convolution kernel.
    param_ranges = init_birdie_param_ranges(start_utc, end_utc)
    birdie_sources = init_birdies(1000, param_ranges)

    do_simulation(start_utc, end_utc, module, sky_array, birdie_sources, 100)
    utils.graph_sky_array(module.sky_band)
    utils.graph_sky_array(sky_array)

print("RUNNING")
main()
print("DONE")
if __name__ == 'main':
    pass
