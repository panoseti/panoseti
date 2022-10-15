"""
Utility functions for the birdie injection program.
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import math
from dateutil import parser
import sys
from functools import cache

sys.path.append('../util')
import pff
import config_file

# Frequently used constants.
ra_range = [0, 360]
dec_range = [-90, 90]


def update_ra_range(ra_left=None, ra_right=None):
    if ra_left:
        ra_range[0] = ra_left
    if ra_right:
        ra_range[1] = ra_right


def update_dec_range(dec_low=None, dec_high=None):
    if dec_low:
        dec_range[0] = dec_low
    if dec_high:
        dec_range[1] = dec_high


def init_ra_dec_ranges(t_start, t_end, initial_bounding_box):
    ra_left = initial_bounding_box[0][0]
    ra_right = initial_bounding_box[0][1] + (t_end - t_start) * 360 / (24 * 60 * 60)
    if abs(ra_right - ra_left) > 360:
        ra_left, ra_right = 0, 360
    update_ra_range(ra_left=ra_left, ra_right=ra_right)
    dec_low = initial_bounding_box[1][0]
    dec_high = initial_bounding_box[1][1]
    update_dec_range(dec_low=dec_low, dec_high=dec_high)
    print(f'ra_range={ra_range}, dec_range={dec_range}')


def get_integration_time(data_dir, run):
    data_config = config_file.get_data_config(f'{data_dir}/{run}')
    x = float(data_config['image']['integration_time_usec'])
    return x


def iso_to_utc(iso_date_string):
    return float(parser.parse(iso_date_string).timestamp())


#@cache
def ra_dec_to_sky_array_indices(ra, dec, bounding_box):
    """Given sky_array, a 2D sky array recording light intensities at RA-DEC coordinates,
    returns the indices in sky_array corresponding to the point (ra, dec).
    ra and dec must be in degrees."""
    ra_index = int(ra_size * ((ra - bounding_box[0][0]) % 360) / ra_length) % ra_size
    dec_index = int(dec_size * ((dec - bounding_box[1][1]) / dec_length)) % dec_size
    return ra_index, dec_index


def get_coord_bounding_box(ra_center, dec_center, r=1.5, pixel_scale=0.31, pixels_per_side=32):
    """Return the ra-dec coordinates of the simulation bounding box centered at ra_center, dec_center.
    """
    interval_radius = pixel_scale * pixels_per_side * (r / 2)
    # Interval of ra coordinates given ra_center, the center of the interval,
    ra_interval = ra_center - interval_radius, ra_center + interval_radius
    # Interval of dec coordinates given dec_center, the center of the interval,
    dec_interval = dec_center - interval_radius, dec_center + interval_radius
    return ra_interval, dec_interval


def init_sky_array_constants(elem_per_deg):
    """r is the ratio of interval length to the module's fov width."""
    global ra_length, ra_size, dec_length, dec_size
    bounding_box = get_coord_bounding_box(0, 0)
    ra_length = bounding_box[0][1] - bounding_box[0][0]
    dec_length = bounding_box[1][1] - bounding_box[1][0]
    ra_size = round(elem_per_deg * ra_length)
    dec_size = round(elem_per_deg * dec_length)


def get_sky_image_array(elem_per_deg, verbose=False):
    """Returns a 2D array with shape (num_ra, num_dec)."""
    init_sky_array_constants(elem_per_deg)
    # 1st dim: RA coords, 2nd dim: DEC coords (both in degrees)
    array_shape = ra_size, dec_size
    if verbose:
        print(f'Array elements per:\n'
              f'\tdeg ra: {round(ra_size / ra_length, 4):<10}\tdeg dec: {round(dec_size / dec_length, 4):<10}')
        print(f'Array shape: {array_shape}, number of elements = {array_shape[0] * array_shape[1]:,}')
    return np.zeros(array_shape)


def graph_sky_array(sky_array):
    """Plot sky_array, labeled with the appropriate RA and DEC ranges."""
    fig, ax = plt.subplots()
    if sky_array is None:
        print('No data to graph.')
        return
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

def line(x0, y0, x1, y1, pts):
    """"Bresenham's line algorithm:
    https://circuitcellar.com/resources/bresenhams-algorithm
    """
    dx = x1 - x0 if x1 >= x0 else x0 - x1
    dy = y0 - y1 if y1 >= y0 else y1 - y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y1 >= y0 else -1
    err = dx + dy
    x = x0
    y = y0
    #print(f'dx, dy, sx, sy, err, x, y={dx, dy, sx, sy, err, x, y}')
    #print(f'x0, y0, x1, y1={x0, y0, x1, y1}')
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

