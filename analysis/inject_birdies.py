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
import time
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

from BirdieSource import BaseBirdieSource
from ModuleView import ModuleView
import birdie_injection_utils as utils

sys.path.append('../util')
import pff
sys.path.append('../control')
import config_file

np.random.seed(383)


def init_birdie_param_ranges(start_utc, end_utc):
    param_ranges = {
        'ra': (180, 360),
        'dec': utils.dec_bounds,
        'file_time_range': (start_utc, end_utc),
        'duty_cycle': (0.25, .75),#(1e-3, 1e-1),
        'period': (1, 4),
        'intensity': (100, 150),
    }
    return param_ranges


def get_birdie_config_vector(param_ranges):
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
    m1 = ModuleView(42, start_utc, 10.3, 44.2, 234, 77, 77, 77)
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

def apply_psf(sky_array, kernel):
    return convolve_fft(sky_array, kernel, allow_huge=False)


def main():
    start_utc = 1656443180
    end_utc = start_utc + 2 * 3600

    # Init ModuleView object
    module = init_module(start_utc)

    # Generate sky array
    utils.reduce_dec_range(module)
    sky_array = init_sky_array(3600*5)

    # Init birdies and convolution kernel.
    param_ranges = init_birdie_param_ranges(start_utc, end_utc)
    birdie_sources = init_birdies(1000, param_ranges)
    kernel = Gaussian2DKernel(x_stddev=4)

    # Move module test
    total_time = 0
    max_counter = 0
    num_reps = 100
    i = 0
    step = (end_utc - start_utc) // num_reps
    t = start_utc
    print('Start simulation')
    while t < end_utc:
        if math.fmod(i, num_reps / 50) < 1:
            print(f'{100 * round(i / num_reps, 3)}%')
            module.plot_simulated_image()
        s = time.time()

        # Update sky
        sky_array.fill(0)
        update_birdies(t, sky_array, birdie_sources)
        convolved_sky_array = apply_psf(sky_array, kernel)

        # Simulate image mode data
        module.update_center_ra_dec_coords(t)
        module.simulate_all_pixel_fovs(convolved_sky_array)

        e = time.time()
        total_time += e - s
        max_counter = max(max_counter, max(module.simulated_img_arr))

        t += step
        i += 1

    avg_time = total_time / num_reps
    print(f'Num sims = {num_reps}, avg sim time = {avg_time} s, total sim time = {total_time} s')
    print(f'max_counter = {max_counter}')
    utils.graph_sky_array(module.sky_band)

print("RUNNING")
main()
print("DONE")
if __name__ == 'main':
    pass
