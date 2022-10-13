"""
Utility functions for calculating the RA-DEC coordinates of a module's field of view, given
an orientation and a unix timestamp.
"""
import numpy as np
import astropy.time
import astropy.units
import astropy.coordinates
from functools import cache


def get_module_center_ra_dec(t, azimuth, elevation, obslat, obslon, obsalt):
    """Return the RA-DEC coordinates of the center of the module's field of view at time t.
    t should be a unix timestamp."""
    angle_unit = astropy.units.deg
    earth_loc = astropy.coordinates.EarthLocation(
        lat=obslat * angle_unit,
        lon=obslon * angle_unit,
        height=obsalt * astropy.units.m,
        ellipsoid='WGS84'
    )
    alt_alz_coords = astropy.coordinates.SkyCoord(
        az=azimuth * angle_unit,
        alt=elevation * angle_unit,
        location=earth_loc,
        obstime=astropy.time.Time(t, format='unix'),
        frame=astropy.coordinates.AltAz
    )
    ra_dec_coords = alt_alz_coords.transform_to('icrs')

    center_ra = ra_dec_coords.ra.value
    center_dec = ra_dec_coords.dec.value
    return center_ra, center_dec


def get_pixel_corner_coord_ftn(center_ra, center_dec, pixel_size=0.31):
    """Output a function which returns the RA-DEC coordinate of
    the corner (x,y) = (0..1, 0..1) of pixel (row, col) = (0..31, 0..31).
    Pixels and corner positions are zero-indexed from the top left."""
    corner_indices = np.linspace(-16, 16, 33)
    ra_offsets = corner_indices * pixel_size
    dec_offsets = np.flip(ra_offsets)

    corner_coords_ra = center_ra + ra_offsets
    corner_coords_dec = center_dec + dec_offsets

    def get_pixel_corner_coord(row, col, x, y):
        return corner_coords_ra[col + y], corner_coords_dec[row + x]
    return get_pixel_corner_coord


def get_module_corner_coords(center_ra, center_dec):
    """Return a 2x2 list containing the RA-DEC coordinates of the corners of a module's FoV."""
    get_pixel_coord = get_pixel_corner_coord_ftn(center_ra, center_dec)
    corner_coords = []
    for i in range(2):
        row = []
        for j in range(2):
            coord = get_pixel_coord(i * 31, j * 31, i, j)
            row.append(coord)
        corner_coords.append(row)
    return corner_coords


def get_sky_band_corner_coords(t_start, t_end, azimuth, elevation, obslat, obslon, obsalt):
    """Return a 2x2 list of RA-DEC coordinates bounding the region of sky observed by a module
    between t_start and t_end. The positions are zero-indexed from the top left corner."""
    assert t_end >= t_start, 'End time cannot be before start time.'
    center_start_ra, center_start_dec = get_module_center_ra_dec(t_start, azimuth, elevation, obslat, obslon, obsalt)
    start_corner_coords = get_module_corner_coords(center_start_ra, center_start_dec)

    center_end_ra, center_end_dec = get_module_center_ra_dec(t_end, azimuth, elevation, obslat, obslon, obsalt)
    end_corner_coords = get_module_corner_coords(center_end_ra, center_end_dec)
    sky_band_corner_coords = [
        [start_corner_coords[0][0], end_corner_coords[0][1]],
        [start_corner_coords[1][0], end_corner_coords[1][1]]
    ]
    return sky_band_corner_coords


#print(get_sky_band_corner_coords(1665164683, 1665225883, 77, 77, 37.3414, 121.64292, 234))
