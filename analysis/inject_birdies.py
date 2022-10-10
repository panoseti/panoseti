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
        - Open a file object for the image file.
    - Main loop
        - Check if we’ve reached EOF in any of the image mode files.
        - Simulate module image mode output.
        - Update image frames (if applicable).
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

np.random.seed(300)

# File IO
def get_birdie_config(birdie_config_path):
    if os.path.exists(birdie_config_path):
        with open(birdie_config_path, 'r+') as f:
            birdie_config = json.loads(f.read())
            param_ranges = birdie_config['param_ranges']
            main()



def update_birdie_log(birdie_log_path):
    """Updates the birdie_log.json file (creating it if necessary) with metadata about each birdie."""
    new_log_data = {
    }
    if os.path.exists(birdie_log_path):
        with open(birdie_log_path, 'r+') as f:
            s = f.read()
            c = json.loads(s)
            # Add new birdie log data
            # Example below
            new_log_data["backup_number"] = c["backups"][-1]["backup_number"] + 1
            c["backups"].append(new_log_data)

            json_obj = json.dumps(c, indent=4)
            f.seek(0)
            f.write(json_obj)
    else:
        c = {
            "backups": [
                new_log_data
            ]
        }
        with open(birdie_log_path, 'w+') as f:
            json_obj = json.dumps(c, indent=4)
            f.write(json_obj)


# Object initialization


def init_module(start_utc):
    m1 = ModuleView('test', start_utc, 10.3, 44.2, 234, 77, 77, 77)
    return m1


def init_sky_array(array_resolution):
    return utils.get_sky_image_array(array_resolution, verbose=True)


def get_birdie_config_vector(param_ranges):
    """Generates a tuple of BirdieSource initialization parameters with uniform distribution
    on the ranges of possible values, provided by param_ranges."""
    unif = np.random.uniform
    config_vector = []
    param_order = ['ra', 'dec', 'start_utc', 'end_utc', 'duty_cycle', 'period', 'intensity']
    for param in param_order:
        config_vector.append(unif(*(param_ranges[param])))
    return config_vector


def init_birdies(num, param_ranges):
    """Initialize BirdieSource objects with randomly selected parameter values
     and store them in a hashmap indexed by RA."""
    birdie_sources = {d: [] for d in range(360)}
    for x in range(num):
        config_vector = get_birdie_config_vector(param_ranges)
        b = BaseBirdieSource(*config_vector)
        birdie_sources[int(b.ra)].append(b)
    return birdie_sources


def init_birdie_param_ranges(start_utc, end_utc):
    """Param_ranges specifies the range of possible values for each BirdieSource parameter."""
    l_ra, r_ra = utils.ra_bounds
    if r_ra < l_ra:
        r_ra += 360
    param_ranges = {
        'ra': (l_ra, r_ra),
        'dec': utils.dec_bounds,
        'start_utc': (start_utc, end_utc),#start_utc),
        'end_utc': (end_utc, end_utc),
        'duty_cycle': (0.1, 1),#(1e-3, 1e-1),
        'period': (0.1, 10),
        'intensity': (100, 400),
    }
    return param_ranges


# Birdie Simulation


def update_birdies(frame_utc, center_ra, sky_array, birdie_sources, pixels_per_side=32, pixel_scale=0.31):
    """Call the generate_birdie method on every BirdieSource object with an RA
    that may be visible by the given module."""
    birdies_in_view = False
    left = int(center_ra - (pixels_per_side // 2) * pixel_scale * 1.1)
    right = int(center_ra + (pixels_per_side // 2) * pixel_scale * 1.1)
    if right < left:
        right += 360
    i = left
    while i <= right:
        for b in birdie_sources[i % 360]:
            point_added = b.generate_birdie(frame_utc, sky_array)
            birdies_in_view = birdies_in_view or point_added
        i += 1
    return birdies_in_view


def apply_psf(sky_array, sigma):
    """Apply a 2d gaussian filter to simulate optical distortion."""
    return gaussian_filter(sky_array, sigma=sigma)


def do_simulation(start_utc, end_utc, module, sky_array, birdie_sources, integration_time, noise_mean, psf_sigma, num_updates=10, plot_images=False, draw_sky_band=False):
    num_steps = math.ceil((end_utc - start_utc) / integration_time)
    time_step = (end_utc - start_utc) / num_steps
    step_num = 0
    print(f'Start simulation of {round((end_utc - start_utc) / 60, 2)} minute file ({num_steps} steps)'
          f'\n\tEstimated time to completion: {round(0.07 * num_steps // 60)} min {round(0.07 * num_steps % 60)} s')
    total_time = max_counter = 0
    s = time.time()
    t = start_utc
    while t < end_utc:
        noisy_img = np.random.poisson(noise_mean, 1024)
        utils.show_progress(step_num, noisy_img, module, num_steps, num_updates, plot_images)

        # Zero sky array.
        sky_array.fill(0)
        # Update module on-sky position.
        module.update_center_ra_dec_coords(t)
        # Update birdie signal points.
        center_ra = module.center_ra
        birdies_in_view = update_birdies(t, center_ra, sky_array, birdie_sources)

        # Sigma is currently an arbitrary value.
        blurred_sky_array = apply_psf(sky_array, sigma=psf_sigma)

        # Simulate image mode data
        module.simulate_all_pixel_fovs(blurred_sky_array, birdies_in_view, draw_sky_band)

        max_counter = max(max_counter, max(module.simulated_img_arr))
        t += time_step
        step_num += 1
    e = time.time()
    total_time += e - s
    avg_time = total_time / num_steps
    print(f'\nNum sims = {num_steps}, avg sim time = {round(avg_time, 5)}s, total sim time = {round(total_time, 4)}s')
    print(f'Max image counter value = {max_counter}')


def do_setup(start_utc, end_utc, num_birdies, arr_res):
    """Initialize objects and arrays for birdie injection."""
    # Init ModuleView object
    module = init_module(start_utc)

    # Limit the simulation to relevant RA-DEC ranges.
    utils.reduce_ra_range(module, start_utc, end_utc)
    utils.reduce_dec_range(module)

    # Init array modeling the sky
    sky_array = init_sky_array(arr_res)

    # Init birdies and convolution kernel.
    param_ranges = init_birdie_param_ranges(start_utc, end_utc)
    birdie_sources = init_birdies(num_birdies, param_ranges)

    return module, sky_array, birdie_sources


def main():
    start_utc = 1685417643
    end_utc = start_utc + 3600
    num_birdies = 20
    integration_time = 20e-1
    # Number of pixels per degree RA and degree DEC
    array_resolution = 50
    psf_sigma = 6
    noise_mean = 0

    module, sky_array, birdie_sources = do_setup(
        start_utc, end_utc, num_birdies, array_resolution
    )

    do_simulation(
        start_utc, end_utc, module, sky_array, birdie_sources,
        integration_time, noise_mean, psf_sigma,
        num_updates=20, plot_images=True, draw_sky_band=True
    )

    # Plot a heatmap of the sky covered during the simulation.
    module.plot_sky_band()
    plt.close()


if __name__ == '__main__':
    print("RUNNING")
    main()
    print("DONE")
