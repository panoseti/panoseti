"""
Utility functions for the birdie injection program.
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import math
from dateutil import parser
import json
import sys
from ModuleView import ModuleView

sys.path.append('../util')
import pff
import config_file

# Possible RA and DEC values in this simulation.
ra_bounds = 0, 360
dec_bounds = -90, 90


def get_integration_time(data_dir, run):
    data_config = config_file.get_data_config(f'{data_dir}/{run}')
    x = float(data_config['image']['integration_time_usec'])
    return x


def iso_to_utc(iso_date_string):
    return float(parser.parse(iso_date_string).timestamp())


def ra_dec_to_sky_array_indices(ra, dec, sky_array):
    """
    Given sky_array, a 2D sky array recording light intensities at RA-DEC coordinates,
    returns the indices in sky_array corresponding to the point (ra, dec).
    ra and dec must be in degrees.
    """
    assert dec_bounds[0] <= dec <= dec_bounds[1], f'lower dec bound = {dec_bounds[0]}, upper dec bound = {dec_bounds[1]}, given dec: {dec}'
    assert dec_bounds[0] <= dec <= dec_bounds[1], f'lower dec bound = {dec_bounds[0]}, upper dec bound = {dec_bounds[1]}, given dec: {dec}'
    shape = np.shape(sky_array)
    ra_size, dec_size = shape[0], shape[1]
    dist = (ra_bounds[1] - ra_bounds[0]) % 360
    if dist == 0:
        dist = 360
    ra_index = int(ra_size * ((ra - ra_bounds[0]) % 360 / dist)) % shape[0]
    dec_index = int(dec_size * ((dec - dec_bounds[0]) / (dec_bounds[1] - dec_bounds[0])))
    return ra_index, dec_index


def reduce_ra_range(mod: ModuleView, start_utc, end_utc):
    global ra_bounds
    # RA of module center at start_utc
    start_ra = mod.center_ra
    end_ra = mod.get_module_ra_at_time(end_utc)
    # Ratio of reduced ra range length to module fov width
    r = 1.2
    margin = mod.pixel_scale * mod.pixels_per_side * (r / 2)
    lower_bound = start_ra - margin
    upper_bound = end_ra + margin
    if abs(lower_bound - upper_bound) < 360:
        ra_bounds = lower_bound % 360, upper_bound % 360
    print(f'Right ascension bounds = ({round(ra_bounds[0], 2)}, {round(ra_bounds[1], 2)}) <deg> '
          f'or ({round(24 * ra_bounds[0] / 360, 2)}, {round(24 * ra_bounds[1] / 360, 2)}) <hr>')
    return ra_bounds


def reduce_dec_range(mod: ModuleView):
    global dec_bounds
    # Ratio of reduced dec range length to module fov width
    r = 1.2
    margin = mod.pixel_scale * mod.pixels_per_side * (r / 2)
    center_dec = mod.center_dec
    lower_bound = max(-90, center_dec - margin)
    upper_bound = min(90, center_dec + margin)
    dec_bounds = lower_bound, upper_bound
    print(f'Declination bounds = ({round(dec_bounds[0], 2)}, {round(dec_bounds[1], 2)}) <deg>')
    return dec_bounds


def get_sky_image_array(elem_per_deg, verbose=False):
    """Returns a 2D array with shape (num_ra, num_dec)."""
    dist = (ra_bounds[1] - ra_bounds[0]) % 360
    if dist == 0:
        dist = 360
    ra_size = round(elem_per_deg * dist)
    dec_size = round(elem_per_deg * (dec_bounds[1] - dec_bounds[0]))
    # 1st dim: RA coords, 2nd dim: DEC coords (both in degrees)
    array_shape = ra_size, dec_size
    if verbose:
        print(f'Array elements per:\n'
              f'\tdeg ra: {round(ra_size / dist, 4):<10}\tdeg dec: {round(dec_size / (dec_bounds[1] - dec_bounds[0]), 4):<10}')
        print(f'Array shape: {array_shape}, number of elements = {array_shape[0] * array_shape[1]:,}')
    return np.zeros(array_shape)


def graph_sky_array(sky_array):
    """Plot sky_array, labeled with the appropriate RA and DEC ranges."""
    fig, ax = plt.subplots()
    if sky_array is None:
        print('No data to graph.')
        return
    # Add RA tick marks
    dist = (ra_bounds[1] - ra_bounds[0]) % 360
    if dist == 0:
        ra_labels_in_deg = np.linspace(0, 360, 7, dtype=np.int)
    else:
        ra_labels_in_deg = (np.linspace(0, dist, 7, dtype=np.int) + ra_bounds[0]) % 360
    ra_labels_in_hrs = 24 * ra_labels_in_deg / 360
    ax.set_xticks(
        np.linspace(0, sky_array.shape[0] - 1, 7),
        ra_labels_in_hrs.round(3)
    )
    # Add DEC tick marks
    ax.set_yticks(
        np.linspace(0, sky_array.shape[1] - 1, 3),
        np.linspace(dec_bounds[0], dec_bounds[1], 3).round(2)
    )
    ax.set_xlabel("Right Ascension")
    ax.set_ylabel("Declination")
    ax.pcolormesh(np.matrix.transpose(sky_array))
    plt.show()
    plt.close(fig)


def show_progress(step_num, img, module, num_steps, num_updates, plot_images=False):
    if step_num % (num_steps // num_updates) == 0:
        v = math.ceil(100 * step_num / num_steps)
        print(f'\tProgress: {v:<2}% [{"*" * (v // 5) + "-" * (20 - (v // 5)):<20}]', end='\r')
        if plot_images and step_num != 0:
            time.sleep(0.01)
            module.plot_simulated_image(img)


def ra_to_degrees(ra):
    """Returns the degree equivalent of a right ascension coordinate
    in the form: (hours, minutes, seconds)."""
    assert len(ra) == 3
    hours = ra[0] + (ra[1] / 60) + (ra[2] / 3600)
    # Rotate 360 degrees in 24 hours.
    degrees = hours * (360 / 24)
    return degrees


def dec_to_degrees(dec):
    """Returns the degree equivalent of a declination coordinate
    in the form: (degrees, arcminutes, arcseconds)."""
    assert len(dec) == 3
    # 60 arcminutes in 1 degree, 3600 arcseconds in 1 degree.
    abs_degrees = abs(dec[0]) + (dec[1] / 60) + (dec[2] / 3600)
    sign = dec[0] / abs(dec[0])
    return sign * abs_degrees

