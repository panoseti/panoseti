import time

import numpy as np
import math
import json
import os
import sys
from scipy.ndimage import gaussian_filter

import birdie_utils
from birdie_source import BaseBirdieSource
from module_view import ModuleView

sys.path.append('../util')
from pff import time_seek, read_image, read_json, img_header_time, write_image_1D

# Setup / Initialization


def init_sky_array(array_resolution):
    return birdie_utils.get_sky_image_array(array_resolution, verbose=True)


def init_module(start_utc, obs_config, module_id, bytes_per_pixel, sky_array):
    if obs_config:
        for dome in obs_config['domes']:
            for module in dome['modules']:
                if module['id'] == module_id:
                    return ModuleView(
                        module_id,
                        start_utc,
                        dome['obslat'],
                        dome['obslon'],
                        dome['obsalt'],
                        module['azimuth'],
                        module['elevation'],
                        module['position_angle'],
                        bytes_per_pixel,
                        sky_array
                    )
    else:
        return ModuleView(
            module_id,
            start_utc,
            37.3425,
            -121.63777,
            1283,
            184.29,
            78.506,
            0,
            sky_array
        )


def init_birdies(num, param_ranges):
    """Initialize BirdieSource objects with randomly selected parameter values
     and store them in a hashmap indexed by RA."""
    birdie_sources = {d: [] for d in range(360)}
    for x in range(num):
        config_vector = birdie_utils.get_birdie_config_vector(param_ranges)
        b = BaseBirdieSource(*config_vector)
        birdie_sources[int(b.ra)].append(b)
    return birdie_sources


def do_setup(start_utc, end_utc, obs_config, birdie_config, bytes_per_pixel, integration_time, module_id):
    """Initialize objects and arrays for birdie injection.
    integration_time is in usec. birdie_config is a file object."""
    print('Setup simulation:')
    # Init array modeling the sky
    sky_array = init_sky_array(birdie_config['array_resolution'])

    # Init ModuleView object
    mod = init_module(start_utc, obs_config, module_id, bytes_per_pixel, sky_array)
    initial_bounding_box = birdie_utils.get_coord_bounding_box(mod.center_ra, mod.center_dec)
    birdie_utils.init_ra_dec_ranges(start_utc, end_utc, initial_bounding_box, module_id)

    # Init birdies and convolution kernel.
    param_ranges = birdie_utils.init_birdie_param_ranges(
        start_utc, end_utc, birdie_config['param_ranges'], module_id
    )
    birdie_sources = init_birdies(birdie_config['num_birdies'], param_ranges)
    sigma = birdie_config['psf_sigma']
    time_step = 1e-6 * integration_time
    return mod, sky_array, birdie_sources, sigma, time_step


# Simulation loop routines


def get_birdie_sources_in_view(frame_utc, bounding_box, birdie_sources):
    birdie_sources_in_view = []
    left = int(bounding_box[0][0] % 360)
    right = int(bounding_box[0][1] % 360) - 1
    if right < left:
        right += 360
    i = left
    while i < right:
        for b in birdie_sources[i % 360]:
            if b.is_in_view(frame_utc):
                birdie_sources_in_view.append(b)
        i += 1
    return birdie_sources_in_view


def update_birdies(frame_utc, bounding_box, sky_array, birdie_sources_in_view):
    """Call the generate_birdie method on every BirdieSource object with an RA
    that may be visible by the given module."""
    for b in birdie_sources_in_view:
        point_added = b.generate_birdie(frame_utc, sky_array, bounding_box)


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


def get_next_frame(file_obj, bytes_per_pixel):
    """Returns the next image frame from file_obj."""
    try:
        j = json.loads(read_json(file_obj))
        img = read_image(file_obj, 32, bytes_per_pixel)
    except Exception as e:
        # Deal with EOF issue in pff.read_json
        if str(e) == "('read_json(): expected {, got', b'')":
            return None, None
        else:
            raise
    return img, j


# Simulation loop.


def do_simulation(data_dir,
                  birdie_dir,
                  start_t,
                  end_t,
                  obs_config,
                  birdie_config,
                  bytes_per_pixel,
                  integration_time,
                  f,
                  nframes,
                  num_updates=20,
                  module_id='test',
                  plot_images=False
                  ):
    # Setup simulation.
    module, sky_array, birdie_sources, sigma, time_step = do_setup(
        start_t, end_t, obs_config, birdie_config, bytes_per_pixel, integration_time, module_id
    )
    frame_size = bytes_per_pixel * 1024 + 1
    avg_t_per_frame = 0.001
    print(f'Start simulation of {round((end_t - start_t) / 60, 2)} minute file ({nframes} frames)'
          f'\n\tEstimated time to completion: {round(avg_t_per_frame * nframes // 60)} '
          f'min {round(avg_t_per_frame * nframes % 60)} s')
    frame_num = 0
    s = time.time()

    while True:
        img, j = get_next_frame(f, bytes_per_pixel)
        if img is None:
            print('\n\tReached EOF.')
            break
        t = img_header_time(j)
        if t > end_t:
            print('\n\tReached last frame in specified range (ok).')
            break

        birdie_utils.show_progress(frame_num, img, module, nframes, num_updates, plot_images)

        # Update module on-sky position.
        module.update_center_ra_dec_coords(t)
        bounding_box = birdie_utils.get_coord_bounding_box(module.center_ra, module.center_dec)

        # Check if any birdies are visible by the module, and
        birdie_sources_in_view = get_birdie_sources_in_view(t, bounding_box, birdie_sources)
        if birdie_sources_in_view:
            # Update birdie signal points.
            sky_array.fill(0)
            update_birdies(t, bounding_box, sky_array, birdie_sources_in_view)
            # Apply a 2d gaussian filter to simulate optical distortion due to the Fesnel lens.
            blurred_sky_array = gaussian_filter(sky_array, sigma=sigma)
            # We must copy the filtered array because the views initialized in module
            # are linked to the original sky_array for efficiency purposes.
            np.copyto(sky_array, blurred_sky_array)
            module.simulate_all_pixel_fovs()
            # Add simulated image to img.
            raw_plus_birdie_img = module.add_birdies_to_image_array(img)
            f.seek(-frame_size, 1)
            write_image_1D(f, raw_plus_birdie_img, 32, bytes_per_pixel)
        frame_num += 1
    e = time.time()
    total_time = e - s
    avg_time = total_time / nframes
    print(f'\nNum sims = {nframes}, avg sim time = {round(avg_time, 5)}s, total sim time = {round(total_time, 4)}s')
    if plot_images:
        birdie_utils.build_gif(data_dir, birdie_dir)
