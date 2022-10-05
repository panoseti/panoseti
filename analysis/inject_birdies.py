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

np.random.seed(383)


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


def init_sky_array(array_resolution):
    return utils.get_sky_image_array(array_resolution)


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
        'duty_cycle': (1, 1),#(1e-3, 1e-1),
        'period': (1, 4),
        'intensity': (100, 150),
    }
    return param_ranges


def do_simulation(start_utc, end_utc, module, sky_array, birdie_sources, integration_time):
    num_steps = math.ceil((end_utc - start_utc) / integration_time)
    run_sim = input(f'Run simulation with {num_steps} steps? Y / N: ')
    if not run_sim:
        return
    total_time = max_counter = 0
    step = (end_utc - start_utc) // num_steps
    t = start_utc
    print('Start simulation')
    while t < end_utc:
        utils.show_progress(t, start_utc, end_utc, step, module)
        s = time.time()
        # Update sky
        sky_array.fill(0)
        update_birdies(t, sky_array, birdie_sources)
        # Sigma = 2 is currently an arbitrary value.
        blurred_sky_array = apply_psf(sky_array, sigma=2)

        # Simulate image mode data
        module.update_center_ra_dec_coords(t)
        module.simulate_all_pixel_fovs(blurred_sky_array, draw_sky_band=True)
        e = time.time()
        total_time += e - s
        max_counter = max(max_counter, max(module.simulated_img_arr))
        t += step
    avg_time = total_time / num_steps
    print(f'\nNum sims = {num_steps}, avg sim time = {round(avg_time, 5)}s, total sim time = {round(total_time, 4)}s')
    print(f'Max image counter value = {max_counter}')


def main():
    # File time interval
    start_utc = 1666456180
    end_utc = start_utc + 2 * 3600
    # Number of pixels per degree RA and degree DEC
    array_resolution = 2
    # Number of birdies to generate
    num_birdies = 1000
    # Number of frames (in seconds)
    integration_time = 20

    # Init ModuleView object
    module = init_module(start_utc)

    # Limit the simulation to relevant RA-DEC ranges.
    utils.reduce_ra_range(module, start_utc, end_utc)
    utils.reduce_dec_range(module)
    sky_array = init_sky_array(array_resolution)
    # Init birdies and convolution kernel.
    param_ranges = init_birdie_param_ranges(start_utc, end_utc)
    birdie_sources = init_birdies(num_birdies, param_ranges)

    do_simulation(start_utc, end_utc, module, sky_array, birdie_sources, integration_time)
    utils.graph_sky_array(module.sky_band)
    utils.graph_sky_array(sky_array)

print("RUNNING")
main()
print("DONE")
if __name__ == 'main':
    pass
