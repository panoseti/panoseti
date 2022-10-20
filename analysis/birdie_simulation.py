import time

import numpy as np
import math
import json
import os
from scipy.ndimage import gaussian_filter

import birdie_utils
from birdie_source import BaseBirdieSource
from module_view import ModuleView


# Setup / Initialization


def init_sky_array(array_resolution):
    return birdie_utils.get_sky_image_array(array_resolution, verbose=True)


def init_module(sky_array, module_id, start_utc):
    m1 = ModuleView(module_id, start_utc, 37.3425, -121.63777, 1283, 184.29, 78.506, 0, sky_array)
    return m1


def init_birdies(num, param_ranges):
    """Initialize BirdieSource objects with randomly selected parameter values
     and store them in a hashmap indexed by RA."""
    birdie_sources = {d: [] for d in range(360)}
    for x in range(num):
        config_vector = birdie_utils.get_birdie_config_vector(param_ranges)
        b = BaseBirdieSource(*config_vector)
        birdie_sources[int(b.ra)].append(b)
    return birdie_sources


def do_setup(start_utc, end_utc, integration_time, birdie_config, module_id='test'):
    """Initialize objects and arrays for birdie injection.
    integration_time is in usec. birdie_config is a file object."""

    # Init array modeling the sky
    sky_array = init_sky_array(birdie_config['array_resolution'])

    # Init ModuleView object
    mod = init_module(sky_array, module_id, start_utc)
    initial_bounding_box = birdie_utils.get_coord_bounding_box(mod.center_ra, mod.center_dec)
    birdie_utils.init_ra_dec_ranges(start_utc, end_utc, initial_bounding_box, module_id)

    # Init birdies and convolution kernel.
    param_ranges = birdie_utils.init_birdie_param_ranges(
        start_utc, end_utc, birdie_config['param_ranges'], module_id
    )
    birdie_sources = init_birdies(birdie_config['num_birdies'], param_ranges)
    sigma = birdie_config['psf_sigma']
    time_step = 1e-6 * integration_time
    num_frames = (end_utc - start_utc) / time_step
    return mod, sky_array, birdie_sources, sigma, num_frames, time_step


# Simulation loop routines


def update_birdies(frame_utc, bounding_box, sky_array, birdie_sources):
    """Call the generate_birdie method on every BirdieSource object with an RA
    that may be visible by the given module."""
    birdies_in_view = False
    left = int(bounding_box[0][0] % 360)
    right = int(bounding_box[0][1] % 360) - 1
    if right < left:
        right += 360
    i = left
    #input(f'left={left}, right={right}')
    while i < right:
        for b in birdie_sources[i % 360]:
            point_added = b.generate_birdie(frame_utc, sky_array, bounding_box)
            birdies_in_view = birdies_in_view or point_added
        i += 1
    return birdies_in_view


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


def get_next_frame(file_obj):
    """Returns the next image frame from file_obj."""
    j, img = None, None
    start_timestamp = None
    try:
        j = pff.read_json(file_obj)
        j = json.loads(j.encode())
        # For ph files: img size = 16 x 16 and bytes per pixel = 2.
        img = pff.read_image(file_obj, 32, 2)
    except Exception as e:
        # Deal with EOF issue in pff.read_json
        if repr(e)[:26] == "Exception('bad type code',":
            return None
    if not j or not img:
        return None
    return img, j


# Simulation loop.


def do_simulation(start_utc,
                  end_utc,
                  birdie_config,
                  integration_time,
                  fin=None,
                  fout=None,
                  noise_mean=0,
                  num_updates=20,
                  plot_images=False,
                  avg_t_per_frame=0.015):
    # Setup simulation.
    module, sky_array, birdie_sources, sigma, num_frames, time_step = \
        do_setup(start_utc, end_utc, integration_time, birdie_config)

    print(f'Start simulation of {round((end_utc - start_utc) / 60, 2)} minute file ({num_frames} frames)'
          f'\n\tEstimated time to completion: {round(avg_t_per_frame * num_frames // 60)} '
          f'min {round(avg_t_per_frame * num_frames % 60)} s')
    frame_num = 0
    s = time.time()

    noisy_img = np.random.poisson(noise_mean, 1024)
    t = start_utc
    while t < end_utc:
        birdie_utils.show_progress(frame_num, noisy_img, module, num_frames, num_updates, plot_images)

        # Update module on-sky position.
        module.update_center_ra_dec_coords(t)
        bounding_box = birdie_utils.get_coord_bounding_box(module.center_ra, module.center_dec)

        # Update birdie signal points.
        sky_array.fill(0)
        birdies_in_view = update_birdies(t, bounding_box, sky_array, birdie_sources)

        # Check if any birdies are visible by the module, and
        if birdies_in_view:
            # Apply a 2d gaussian filter to simulate optical distortion due to the Fesnel lens.
            blurred_sky_array = gaussian_filter(sky_array, sigma=sigma)
            # We must copy the filtered array because the views initialized in module
            # are linked to the original sky_array for efficiency purposes.
            np.copyto(sky_array, blurred_sky_array)
            module.simulate_all_pixel_fovs()

        t += time_step
        frame_num += 1

    e = time.time()
    total_time = e - s
    avg_time = total_time / num_frames
    print(f'\nNum sims = {num_frames}, avg sim time = {round(avg_time, 5)}s, total sim time = {round(total_time, 4)}s')
    if plot_images:
        birdie_utils.build_gif()


#raw_img, j = get_next_frame(fin)
# def write_image_1D(f, img, img_size, bytes_per_pixel):
# pff.write_image_1D(fout, module.add_birdies_to_image_array(noisy_img), 32, 2)

def simulation_dispatch():
    pass