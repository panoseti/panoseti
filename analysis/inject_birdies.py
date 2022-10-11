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


import analysis_util
from BirdieSource import BaseBirdieSource
from ModuleView import ModuleView
import birdie_injection_utils as birdie_utils

sys.path.append('../util')
import pff
sys.path.append('../control')
import config_file

np.random.seed(300)

# File IO

DATA_DIR = 'nico/downloads/test_data/data/obs_Lick.start_2022-05-11T23/38/29Z.runtype_eng.pffd'
fname = 'start_2022-05-11T23/39/15Z.dp_1.bpp_2.dome_0.module_1.seqno_1.pff'


def get_birdie_config(birdie_config_path):
    """Loads a birdie injection config file with the form:
    {
        "integration_time": 20e-1,
        "num_birdies": 30,
        "array_resolution": 50,
        "psf_sigma": 6,
        "param_ranges": {
            "duty_cycle": [0.1, 1],
            "period": [1, 10],
            "intensity": [100, 400]
        }
    }
    array_resolution: number of pixels per degree RA and degree DEC.
    psf_sigma: value of sigma used in the simulated (gaussian) point-spread function.
    param_ranges: possible values for BirdieSource objects.
    """
    if os.path.exists(birdie_config_path):
        with open(birdie_config_path, 'r+') as f:
            birdie_config = json.loads(f.read())
            return birdie_config
    else:
        print(f'{birdie_config_path} is not a valid path.')


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
            'Stuff'
        }
        with open(birdie_log_path, 'w+') as f:
            json_obj = json.dumps(c, indent=4)
            f.write(json_obj)


def get_obs_config(data_dir_path):
    pass


# Object initialization


def init_module(start_utc):
    m1 = ModuleView('test', start_utc, 10.3, 44.2, 234, 77, 77, 77)
    return m1


def init_sky_array(array_resolution):
    return birdie_utils.get_sky_image_array(array_resolution, verbose=True)


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


def init_birdie_param_ranges(start_utc, end_utc, param_ranges):
    """Param_ranges specifies the range of possible values for each BirdieSource parameter."""
    l_ra, r_ra = birdie_utils.ra_bounds
    l_dec, r_dec = birdie_utils.dec_bounds
    if r_ra < l_ra:
        r_ra += 360
    param_ranges['ra'] = (l_ra, r_ra)
    param_ranges['dec'] = (l_dec, r_dec)
    param_ranges['start_utc'] = (start_utc, start_utc)
    param_ranges['end_utc'] = (end_utc, end_utc)
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


def do_simulation(start_utc,
                  end_utc,
                  module,
                  sky_array,
                  birdie_sources,
                  birdie_config,
                  nframes,
                  noise_mean=0,
                  num_updates=10,
                  plot_images=False,
                  draw_sky_band=False):
    time_step = (end_utc - start_utc) / nframes
    step_num = 0
    print(f'Start simulation of {round((end_utc - start_utc) / 60, 2)} minute file ({nframes} steps)'
          f'\n\tEstimated time to completion: {round(0.07 * nframes // 60)} min {round(0.07 * nframes % 60)} s')
    total_time = max_counter = 0
    s = time.time()
    t = start_utc
    while t < end_utc:
        noisy_img = np.random.poisson(noise_mean, 1024)
        birdie_utils.show_progress(step_num, noisy_img, module, nframes, num_updates, plot_images)

        # Update module on-sky position.
        module.update_center_ra_dec_coords(t)

        # Update birdie signal points.
        sky_array.fill(0)
        center_ra = module.center_ra
        birdies_in_view = update_birdies(t, center_ra, sky_array, birdie_sources)

        # Simulate image mode data
        blurred_sky_array = apply_psf(sky_array, sigma=birdie_config['psf_sigma'])
        module.simulate_all_pixel_fovs(blurred_sky_array, birdies_in_view, draw_sky_band)

        max_counter = max(max_counter, max(module.simulated_img_arr))
        t += time_step
        step_num += 1
    e = time.time()
    total_time += e - s
    avg_time = total_time / nframes
    print(f'\nNum sims = {nframes}, avg sim time = {round(avg_time, 5)}s, total sim time = {round(total_time, 4)}s')
    print(f'Max image counter value = {max_counter}')


def do_setup(start_utc, end_utc, birdie_config):
    """Initialize objects and arrays for birdie injection."""

    # Init ModuleView object
    mod = init_module(start_utc)

    # Limit the simulation to relevant RA-DEC ranges.
    birdie_utils.reduce_ra_range(mod, start_utc, end_utc)
    birdie_utils.reduce_dec_range(mod)

    # Init array modeling the sky
    sky_array = init_sky_array(birdie_config['array_resolution'])

    # Init birdies and convolution kernel.
    param_ranges = init_birdie_param_ranges(start_utc, end_utc, birdie_config['param_ranges'])
    birdie_sources = init_birdies(birdie_config['num_birdies'], param_ranges)

    return mod, sky_array, birdie_sources


def main():
    #analysis_dir = analysis_util.make_analysis_dir('birdie_injection', run)

    start_utc = 1685417643
    end_utc = start_utc + 3600
    integration_time = 20e-1
    birdie_config = get_birdie_config('birdie_config.json')

    module, sky_array, birdie_sources = do_setup(start_utc, end_utc, birdie_config)

    nframes = math.ceil((end_utc - start_utc) / integration_time)
    do_simulation(
        start_utc, end_utc,
        module, sky_array, birdie_sources,
        birdie_config,
        nframes,
        noise_mean=0,
        num_updates=20,
        plot_images=True,
        draw_sky_band=True
    )

    # Plot a heatmap of the sky covered during the simulation.
    module.plot_sky_band()
    plt.close()


if __name__ == '__main__':
    print("RUNNING")
    main()
    print("DONE")
