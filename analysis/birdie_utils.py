"""
Utility functions for the birdie injection program.
"""
import time
import math
import json
import sys
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import imageio


sys.path.append('../util')
import pff
import config_file
import analysis_util

# Dicts of simulation-level constants.
module_constants = dict()
sky_arr_consts = dict()


# Interface between RA-DEC coordinates and the sky_array abstraction.


def get_ra_dec_ranges(coord, module_id):
    """Return the possible RA or DEC ranges for this simulation """
    if coord == 'ra':
        return module_constants[module_id]['coord_ranges'][0]
    elif coord == 'dec':
        return module_constants[module_id]['coord_ranges'][1]
    else:
        assert False, f'coord must be equal to either "ra" or "dec", not {coord}.'


def init_ra_dec_ranges(t_start, t_end, initial_bounding_box, module_id, verbose):
    """Initialize the ranges of possible RA and DEC coordinates in the
    simulation for module module_id."""
    if module_id not in module_constants:
        module_constants[module_id] = dict()
    ra_left = initial_bounding_box[0][0]
    ra_right = initial_bounding_box[0][1] + (t_end - t_start) * 360 / (24 * 60 * 60)
    if abs(ra_right - ra_left) > 360:
        ra_left, ra_right = 0, 360
    ra_range = ra_left, ra_right
    dec_low = initial_bounding_box[1][0]
    dec_high = initial_bounding_box[1][1]
    dec_range = dec_low, dec_high
    if verbose: print(f"\t\tmodule '{module_id}': ra_range={round(ra_range[0], 3), round(ra_range[1], 3)} <deg>, "
          f"dec_range={round(dec_range[0], 3), round(dec_range[1], 3)} <deg>")
    module_constants[module_id]['coord_ranges'] = ra_range, dec_range


def ra_dec_to_sky_array_indices(ra, dec, bounding_box):
    """Given sky_array, a 2D sky array recording light intensities at RA-DEC coordinates,
    returns the indices in sky_array corresponding to the point (ra, dec).
    ra and dec must be in degrees."""
    ra_len, dec_len = sky_arr_consts['coord_lens']
    shape = sky_arr_consts['shape']
    ra_index = int(shape[0] * ((ra - bounding_box[0][0]) % 360) / ra_len) % shape[0]
    dec_index = int(shape[1] * ((dec - bounding_box[1][1]) / dec_len)) % shape[1]
    return ra_index, dec_index


def get_coord_bounding_box(ra_center, dec_center, r=1.5, pixel_scale=0.31, pixels_per_side=32):
    """Return the ra-dec coordinates of the simulation bounding box centered at ra_center, dec_center.
    r is the ratio of interval length to the module's fov width.
    """
    assert r >= 1.42, 'r must be at least sqrt 2. Needed to avoid weird results in the case where pos_angle = 45 degrees.'
    interval_radius = pixel_scale * pixels_per_side * (r / 2)
    # Interval of ra coordinates given ra_center, the center of the interval,
    ra_interval = ra_center - interval_radius, ra_center + interval_radius
    # Interval of dec coordinates given dec_center, the center of the interval,
    dec_interval = dec_center - interval_radius, dec_center + interval_radius
    return ra_interval, dec_interval


def init_sky_array_constants(elem_per_deg):
    """ra_len is the length in degrees of RA interval represented by sky_array.
    dec_len is the length in degrees of the interval of DEC coordinates represented by sky_array."""
    bounding_box = get_coord_bounding_box(0, 0)
    ra_len, dec_len = bounding_box[0][1] - bounding_box[0][0], bounding_box[1][1] - bounding_box[1][0]
    sky_arr_consts['coord_lens'] = ra_len, dec_len

    shape = round(elem_per_deg * ra_len), round(elem_per_deg * dec_len)
    sky_arr_consts['shape'] = shape


def get_sky_image_array(elem_per_deg, verbose):
    """Returns a 2D array with shape (num_ra, num_dec)."""
    init_sky_array_constants(elem_per_deg)
    # 1st dim: RA coords, 2nd dim: DEC coords (both in degrees)
    ra_length, dec_length = sky_arr_consts['coord_lens']
    array_shape = sky_arr_consts['shape']
    if verbose:
        print(f'\t\tArray elements per:\n'
              f'\t\t\tdeg ra: {round(array_shape[0] / ra_length, 4):<10}\t\tdeg dec: {round(array_shape[1] / dec_length, 4):<10}')
        print(f'\t\tArray shape: {array_shape}, number of elements = {array_shape[0] * array_shape[1]:,}')
    return np.zeros(array_shape, dtype=np.float32)


def graph_sky_array(sky_array, module_id):
    """Plot sky_array, labeled with the appropriate RA and DEC ranges."""
    fig, ax = plt.subplots()
    if sky_array is None:
        print('No data to graph.')
        return
    ra_range = get_ra_dec_ranges('ra', module_id)
    dec_range = get_ra_dec_ranges('dec', module_id)
    # Add RA tick marks
    dist = (ra_range[1] - ra_range[0]) % 360
    if dist == 0:
        ra_labels_in_deg = np.linspace(0, 360, 7, dtype=np.int)
    else:
        ra_labels_in_deg = (np.linspace(0, dist, 7, dtype=np.int) + ra_range[0]) % 360
    ra_labels_in_hrs = 24 * ra_labels_in_deg / 360
    ax.set_xticks(
        np.linspace(0, sky_array.shape[0] - 1, 7),
        ra_labels_in_deg.round(3)
    )
    # Add DEC tick marks
    ax.set_yticks(
        np.linspace(0, sky_array.shape[1] - 1, 3),
        np.linspace(dec_range[0], dec_range[1], 3).round(2)
    )
    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declination")
    ax.pcolormesh(np.matrix.transpose(sky_array))
    plt.show()
    plt.close(fig)


# Simulation utils

module_plots = dict()


def show_progress(step_num, img, module, num_steps, num_updates, plot_images=False):
    if step_num % (num_steps // num_updates) == 0:
        v = math.ceil(100 * step_num / num_steps)
        print(f'\t\tProgress: {v:<2}% [{"*" * (v // 5) + "-" * (20 - (v // 5)):<20}]', end='\r')
        if plot_images and step_num != 0:
            if module.module_id not in module_plots:
                os.system(f'mkdir -p ./birdie_injection_plots')
                module_plots[module.module_id] = []
            fig = module.plot_simulated_image(img)
            fname = f'./birdie_injection_plots/{time.time()}.png'
            module_plots[module.module_id].append(fname)
            plt.savefig(fname)
            plt.close(fig)


def build_gif(data_dir, birdie_dir, module_id):
    if module_id in module_plots:
        with imageio.get_writer(f'{data_dir}/{birdie_dir}/{module_id}_{time.time()}.gif', mode='I') as writer:
            for fname in module_plots[module_id]:
                image = imageio.imread(fname)
                writer.append_data(image)
            for filename in module_plots[module_id]:
                os.remove(filename)


def bresenham_line(x0, y0, x1, y1, pts):
    """"Bresenham's line algorithm implementation from
    https://circuitcellar.com/resources/bresenhams-algorithm.
    pts is a dictionary of [y-index] : [min_x index, max_x index],
    for scanline rasterization.
    """
    dx = x1 - x0 if x1 >= x0 else x0 - x1
    dy = y0 - y1 if y1 >= y0 else y1 - y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y1 >= y0 else -1
    err = dx + dy
    x = x0
    y = y0

    while True:
        pts[y][0], pts[y][1] = min(pts[y][0], x), max(pts[y][1], x)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy: # step x
            err += dy
            x += sx
        if e2 <= dx: # step y
            err += dx
            y += sy


# Config file IO

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
    config_file.check_config_file(birdie_config_path)
    with open(birdie_config_path, 'r+') as f:
        birdie_config = json.loads(f.read())
        return birdie_config


def get_integration_time(data_dir, run_dir):
    """Get the integration time for this run."""
    data_config = config_file.get_data_config(f'{data_dir}/{run_dir}')
    x = float(data_config['image']['integration_time_usec'])
    return x


# Setup birdie run directory


def get_birdie_sequence_num(data_dir, run_dir, verbose):
    """Get birdie sequence number, equal to current max birdie
    sequence number for run_dir plus 1."""
    max_sequence_num = 0
    run_name_without_pffd = run_dir.replace('.pffd', '')
    for f in os.listdir(f'{data_dir}'):
        if run_name_without_pffd in f.replace('.pffd', ''):
            run_attrs = pff.parse_name(f)
            if 'birdie' in run_attrs:
                max_sequence_num = max(max_sequence_num, int(run_attrs['birdie']))
    if verbose: print(f'\t\tBirdie sequence_num = {max_sequence_num + 1}')
    return max_sequence_num + 1


def check_image_files(data_dir, run_dir, fnames):
    files_ok = []
    for fname in fnames:
        f_path = f'{data_dir}/{run_dir}/{fname}'
        if os.path.exists(f_path):
            file_attrs = pff.parse_name(fname)
            bytes_per_pixel = int(file_attrs['bpp'])
            bytes_per_image = 1 + bytes_per_pixel * 1024
            with open(f_path, 'rb') as fin:
                try:
                    pff.img_info(fin, bytes_per_image)
                except Exception as e:
                    msg = f'"{f_path}" may have variable-length json.\nError message: "{e}"\n'
                    print(msg)
                    files_ok.append(False)
                    continue
                files_ok.append(True)
        else:
            files_ok.append(False)
    return len(files_ok) > 0 and all(files_ok)


def make_birdie_dir(data_dir, run_dir, sequence_num):
    """Create directory for run + birdie data."""
    birdie_dir = run_dir.replace('.pffd', '') + f'.birdie_{sequence_num}.pffd'
    analysis_util.make_dir(f'{data_dir}/{birdie_dir}')
    for fname in os.listdir(f'{data_dir}/{run_dir}'):
        if not pff.is_pff_file(fname):
            shutil.copy(
                f'{data_dir}/{run_dir}/{fname}',
                f'{data_dir}/{birdie_dir}/{fname}'
            )
            continue
        if pff.pff_file_type(fname) not in ('img16', 'img8'):
            continue
    return birdie_dir


def make_birdie_log_files(data_dir, birdie_dir, module_id):
    """Creates birdie log files.
    birdie_log.json stores information about every birdie added to an image frame.
    This file has the format:
    {
        frame unix time: {
            'birdies': [
                Birdie log entries for every birdie added to the frame with timestamp t.
                Each birdie log entry has the form:
                {
                    'birdie_id': hash identifying a birdie source in birdie_sources.json,
                    'intensity': intensity in raw adc of birdie.
                }
            ]
        }
    }
    Since the RA-DEC coordinates of a birdie source are (presumably) static, we only store the birdie source object
    hash and the intensity (which may change as a function of time). To get the static metadata about a particular
    birdie source, we can use key birdie_id in the birdie_sources.json, described below.

    birdie_sources.json stores configuration information about each birdie source used in birdie injection.
    This file has the format:
    {
        birdie source object hash: {
            'class_name': name of subclass used for this birdie source.
            'params': {
                configuration params and their values.
            }
        }
    }
    """
    birdie_log_path = f'{data_dir}/{birdie_dir}/birdie_log.module_{module_id}.json'
    birdie_sources_path = f'{data_dir}/{birdie_dir}/birdie_sources.json'
    if not os.path.exists(birdie_log_path):
        with open(birdie_log_path, 'w'):
            pass
    if not os.path.exists(birdie_sources_path):
        with open(birdie_sources_path, 'w'):
            pass
    return birdie_log_path, birdie_sources_path

