"""
Utility functions for the birdie injection program.
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import ModuleView

dec_bounds = -90, 90

def ra_dec_to_sky_array_indices(ra, dec, sky_array):
    """
    Given sky_array, a 2D sky array recording light intensities at RA-DEC coordinates,
    returns the indices in sky_array corresponding to the point (ra, dec).
    ra and dec must be in degrees.
    """
    #assert dec_bounds[0] <= dec <= dec_bounds[1], f'lower dec bound = {dec_bounds[0]}, upper dec bound = {dec_bounds[1]}, given dec: {dec}'
    shape = np.shape(sky_array)
    ra_size, dec_size = shape[0], shape[1]
    ra_index = math.floor(ra_size * (ra / 360)) % shape[0]
    dec_index = math.floor(dec_size * ((dec - dec_bounds[0]) / (dec_bounds[1] - dec_bounds[0])))
    return ra_index, dec_index


def get_sky_image_array(num_ra, num_dec=None, dtype=np.float16):
    """Returns a 2D array with shape (num_ra, num_dec)."""
    if num_dec is None:
        num_dec = int(num_ra * ((dec_bounds[1] - dec_bounds[0]) / 360))
    # 1st dim: RA coords, 2nd dim: DEC coords (both in degrees)
    array_shape = num_ra, num_dec
    return np.zeros(array_shape)


def reduce_dec_range(mod: ModuleView):
    global dec_bounds
    center_dec = mod.center_dec
    pixel_scale = mod.pixel_scale
    pixels_per_side = mod.pixels_per_side
    lower_bound = max(-90, center_dec - pixel_scale * (0.75 * pixels_per_side))
    upper_bound = min(90, center_dec + pixel_scale * (0.75 * pixels_per_side))
    dec_bounds = lower_bound, upper_bound
    print(f'Declination bounds = {dec_bounds}')
    return dec_bounds


def ra_to_degrees(ra):
    """
    Returns the degree equivalent of a given right ascension coordinate
    ra in the form: (hours, minutes, seconds).
    """
    assert len(ra) == 3
    hours = ra[0] + (ra[1] / 60) + (ra[2] / 3600)
    # Rotate 360 degrees in 24 hours.
    degrees = hours * (360 / 24)
    return degrees


def dec_to_degrees(dec):
    """
    Returns the degree equivalent of a given declination coordinate
    dec in the form: (degrees, arcminutes, arcseconds).
    """
    assert len(dec) == 3
    # 60 arcminutes in 1 degree, 3600 arcseconds in 1 degree.
    abs_degrees = abs(dec[0]) + (dec[1] / 60) + (dec[2] / 3600)
    sign = dec[0] / abs(dec[0])
    return sign * abs_degrees


def graph_sky_array(sky_array):
    if sky_array is None:
        print('No data to graph.')
        return
    plt.yticks(np.linspace(0, sky_array.shape[1] - 1, 2), np.linspace(
        round(dec_bounds[0], 1), round(dec_bounds[1], 1), 2, dtype=float))
    plt.xticks(np.linspace(0, sky_array.shape[0] - 1, 7), np.linspace(0, 24, 7, dtype=np.int))
    plt.xlabel("Right Ascension")
    plt.ylabel("Declination")
    plt.imshow(np.matrix.transpose(sky_array), interpolation='none')
    plt.gca().invert_yaxis()
    plt.show()


def show_progress(t, start_utc, end_utc, step, module, num_updates=10):
    if math.fmod(t - start_utc, (end_utc - start_utc) // (num_updates * step)) < 0.1:
        v = math.ceil(100 * (t - start_utc) / (end_utc - start_utc))
        print(f'\tProgress: {v:<2}% [{"*" * (v // 10):<10}]', end='\r')
        module.plot_simulated_image()

