#! /usr/bin/env python3

"""
Script for calculating the time between the
full width half maximum of a PSF convoluted with a step function.
"""
import math

import numpy as np
from scipy.stats import norm


def get_nearest_index(data, value):
    """
    Returns the index of the element in data with the
    smallest absolute difference from 'value'.
    """
    array = np.asarray(data)
    index = (np.abs(array - value)).argmin()
    return index


def get_intensity_limits(total_intensity):
    """
    Calculates the two intensity levels corresponding to
    half of a star's PSF total_intensity.

    :param total_intensity: total intensity of the star.
    :return: tuple of lower and upper limits.
    """
    rv = norm()
    half_max = math.sqrt(2 * math.log(2))
    lower = total_intensity * (1 - rv.cdf(half_max))
    upper = total_intensity * (rv.cdf(half_max))
    return lower, upper


def get_fwhm_time(total_intensity, data):
    """
    :param total_intensity: total intensity of the star.
    :return: Returns the time between the full width at half maximum of the star's PSF.??
    """
    lower, upper = get_intensity_limits(total_intensity)
    # Get index of data values closest to lower and upper.
    lower_index = get_nearest_index(data, lower)
    upper_index = get_nearest_index(data, upper)
    return upper_index - lower_index
