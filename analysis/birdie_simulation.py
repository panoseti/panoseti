"""
Routines for generating simulated images for birdie injection, including:
    - Module view and birdie source initialization.
    - Birdie metadata recording.
    - Simulated image generation.
    - Combination of simulated image and raw image.
    - Editing pff files with birdie + raw images.
"""

import time
import numpy as np
import json
import sys
from scipy.ndimage import gaussian_filter

import birdie_utils
from birdie_source import BaseBirdieSource
from module_view import ModuleView

sys.path.append('../util')
from pff import time_seek, read_image, read_json, img_header_time, write_image_1D

np.random.seed(42)


# Setup: BirdieSources, ModuleView, and sky array.


def init_sky_array(array_resolution, verbose):
    return birdie_utils.get_sky_image_array(array_resolution, verbose)


def init_module(start_t, obs_config, module_id, bytes_per_pixel, sky_array):
    if obs_config:
        for dome in obs_config['domes']:
            for module in dome['modules']:
                if module['id'] == module_id:
                    return ModuleView(
                        module_id,
                        start_t,
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
            start_t,
            37.3425,
            -121.63777,
            1283,
            184.29,
            78.506,
            0,
            bytes_per_pixel,
            sky_array
        )


def init_birdie_param_ranges(start_t, end_t, param_ranges, module_id):
    """Param_ranges specifies the range of possible values for each BirdieSource parameter."""
    l_ra, r_ra = birdie_utils.get_ra_dec_ranges('ra', module_id)
    l_dec, r_dec = birdie_utils.get_ra_dec_ranges('dec', module_id)
    if r_ra < l_ra:
        r_ra += 360

    param_ranges['ra'] = (l_ra, r_ra)
    param_ranges['dec'] = (l_dec, r_dec)
    param_ranges['start_t'] = (start_t, (start_t + end_t) / 2)
    param_ranges['end_t'] = ((start_t + end_t) / 2, end_t)
    return param_ranges


def get_birdie_source_config(param_ranges):
    """Generates a tuple of BirdieSource initialization parameters with uniform distribution
    on the ranges of possible values, provided by param_ranges."""
    unif = np.random.uniform
    birdie_config = dict()
    param_order = ['ra', 'dec', 'start_t', 'end_t', 'duty_cycle', 'period', 'intensity']
    for param in param_order:
        birdie_config[param] = unif(*(param_ranges[param]))
    birdie_config['ra'] %= 360
    return birdie_config


def init_birdies(num, param_ranges, birdie_sources_path):
    """Initialize BirdieSource objects with randomly selected parameter values
     and store them in a hashmap indexed by RA."""
    birdie_source_metadata = dict()
    birdie_sources = {d: [] for d in range(360)}
    for x in range(num):
        birdie_source_config = get_birdie_source_config(param_ranges)
        b = BaseBirdieSource(birdie_source_config)
        birdie_sources[int(b.config['ra'])].append(b)
        # Save metadata entry for this birdie.
        birdie_source_metadata[hash(b)] = {
            'class_name': type(b).__name__,
            'birdie_config': birdie_source_config
        }
    with open(birdie_sources_path, 'w') as f:
        json.dump(birdie_source_metadata, f, indent=4)
    return birdie_sources


def do_setup(start_t, end_t, obs_config, birdie_config, bytes_per_pixel,
             integration_time, module_id, birdie_sources_path, verbose):
    """Initialize objects and arrays for birdie injection.
    integration_time is in usec. birdie_config is a file object."""
    if verbose: print('\tSimulation setup:')
    # Init array modeling the sky
    sky_array = init_sky_array(birdie_config['array_resolution'], verbose)

    # Init ModuleView object
    mod = init_module(start_t, obs_config, module_id, bytes_per_pixel, sky_array)
    initial_bounding_box = birdie_utils.get_coord_bounding_box(mod.center_ra, mod.center_dec)
    birdie_utils.init_ra_dec_ranges(start_t, end_t, initial_bounding_box, module_id, verbose)

    # Init birdies and convolution kernel.
    param_ranges = init_birdie_param_ranges(
        start_t, end_t, birdie_config['param_ranges'], module_id
    )
    birdie_sources = init_birdies(birdie_config['num_birdies'], param_ranges, birdie_sources_path)
    sigma = birdie_config['psf_sigma']
    return mod, sky_array, birdie_sources, sigma


# Simulation loop routines


def get_birdie_sources_in_view(frame_t, bounding_box, birdie_sources):
    birdie_sources_in_view = []
    left = int(bounding_box[0][0] % 360)
    right = int(bounding_box[0][1] % 360) - 1
    if right < left:
        right += 360
    i = left
    while i < right:
        for b in birdie_sources[i % 360]:
            if b.is_in_view(frame_t):
                birdie_sources_in_view.append(b)
        i += 1
    return birdie_sources_in_view


def update_birdies(frame_t, bounding_box, sky_array, birdie_sources_in_view, birdie_log_dict):
    """Call the generate_birdie method on every BirdieSource object with an RA
    that may be visible by the given module."""
    birdies = []
    for b in birdie_sources_in_view:
        log_entry = b.generate_birdie(frame_t, sky_array, bounding_box)
        birdies.append(log_entry)
    birdie_log_dict[frame_t]['birdies'] = birdies


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
                  birdie_log_path,
                  birdie_sources_path,
                  start_t,
                  end_t,
                  obs_config,
                  birdie_config,
                  module_id,
                  bytes_per_pixel,
                  integration_time,
                  f,
                  bytes_per_image,
                  nframes,
                  verbose,
                  num_updates=20,
                  plot_images=False
                  ):
    """Dispatch function for simulation routines."""
    # Setup simulation.
    module, sky_array, birdie_sources, sigma = do_setup(
        start_t, end_t, obs_config, birdie_config, bytes_per_pixel,
        integration_time, module_id, birdie_sources_path, verbose
    )
    birdie_log_dict = dict()

    print(f'\tStart simulation of {round((end_t - start_t) / 60, 2)} minute file ({round(nframes)} frames):')
    frame_num = 0
    s = time.time()
    while True:
        # Read next json and image array; exiting if they do not exist.
        img, j = get_next_frame(f, bytes_per_pixel)
        if img is None:
            print('\n\t\tReached EOF.')
            break
        t = img_header_time(j)
        if t > end_t:
            print('\n\t\tReached last frame in specified range (ok).')
            break

        birdie_utils.show_progress(frame_num, img, module, nframes, num_updates, plot_images)

        # Update module on-sky position.
        module.update_center_ra_dec_coords(t)
        bounding_box = birdie_utils.get_coord_bounding_box(module.center_ra, module.center_dec)

        # Check if any birdies are visible by the module, and
        birdie_sources_in_view = get_birdie_sources_in_view(t, bounding_box, birdie_sources)
        if birdie_sources_in_view:
            # Clear sky array.
            sky_array.fill(0)
            # Update birdie signal points.
            birdie_log_dict[t] = dict()
            update_birdies(t, bounding_box, sky_array, birdie_sources_in_view, birdie_log_dict)
            # Apply a 2d gaussian filter to simulate optical distortion due to the Fesnel lens.
            blurred_sky_array = gaussian_filter(sky_array, sigma=sigma)
            # We must copy the filtered array because the views initialized in module
            # are linked to the original sky_array for efficiency purposes.
            np.copyto(sky_array, blurred_sky_array)
            module.simulate_all_pixel_fovs()
            # Add simulated image to img.
            raw_plus_birdie_img = module.add_birdies_to_image_array(img)
            f.seek(-bytes_per_image, 1)
            write_image_1D(f, raw_plus_birdie_img, 32, bytes_per_pixel)
        frame_num += 1
    with open(birdie_log_path, 'w') as f:
        json.dump(birdie_log_dict, f, indent=4)
    e = time.time()
    total_time = e - s
    avg_time = total_time / nframes
    #print(f'Number of loops = {nframes}, avg time per loop = {round(avg_time, 5)}s, total time = {round(total_time, 4)}s')
    if plot_images:
        birdie_utils.build_gif(data_dir, birdie_dir, module_id, verbose)
