"""
Utility functions for the birdie injection program.
"""
import numpy as np

def ra_dec_to_img_array_indices(ra, dec, img_array):
    """
    Given img_array, a 3D image array recording light intensities at each RA-DEC coordinate,
    returns the indices in img_array corresponding to the point (ra, dec).
    ra and dec must be in degrees.
    """
    shape = np.shape(img_array)
    assert len(shape) == 3, 'img_array must be a 3D matrix.'
    ra_size, dec_size = shape[0], shape[1]
    ra_index = int(ra_size * (ra / 360))
    dec_index = int(dec_size * ((dec + 90) / 180))
    return ra_index, dec_index


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


ra1 = (6, 45, 9)
dec1 = (-16, 42, 58)

print(ra_to_degrees(ra1))
print(dec_to_degrees(dec1))


