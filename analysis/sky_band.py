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


def get_module_pixel_corner_coord_ftn(pos_angle, pixel_size=0.31):
    """Returns a higher-order function that returns functions that
    return the RA-DEC coordinate of a pixel corner. The environment of
    this function makes module-wide constants available throughout the simulation.
        pos_angle: orientation of the astronomical instr/image on the plane
        of the sky, measured in degrees from North to East"""
    pos_angle = -79
    # Pixel offsets from the center of the module's FoV.
    col_offsets, row_offsets = np.linspace(-16, 16, 33), np.linspace(16, -16, 33)
    #input(f'col_offsets={col_offsets}, \nrow_offsets={row_offsets}')

    # Get the rotation matrix according to the position angle of the module.
    theta = np.radians(-pos_angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(
        ((c, -s),
         (s, c))
    )
    # Basis vectors for coordinate grid positions.
    # (x, y) = (RA coordinate, DEC coordinate)
    i_hat = rotation_matrix.dot(np.array((1, 0))) * pixel_size
    j_hat = rotation_matrix.dot(np.array((0, 1))) * pixel_size

    #input(f'i_hat={i_hat[:, np.newaxis]}, \nj_hat={j_hat[:, np.newaxis]}')

    #input(f'i_hat * col_offsets={col_offsets * i_hat[:, np.newaxis]}')
    #input(f'j_hat * row_offsets={row_offsets * j_hat[:, np.newaxis]}')

    pixel_corner_i_hat_coords = col_offsets * i_hat[:, np.newaxis]
    pixel_corner_j_hat_coords = row_offsets * j_hat[:, np.newaxis]
    #coords = pixel_corner_i_hat_coords + pixel_corner_j_hat_coords
    #input(coords)

    def get_pixel_corner_coord_ftn(center_ra, center_dec):
        """Returns a function that returns the RA-DEC coordinate of
        the corner (x,y) = (0..1, 0..1) of pixel (row, col) = (0..31, 0..31).
        Pixels and corner positions are zero-indexed from the top left."""
        corner_coords_ra = center_ra + pixel_corner_i_hat_coords[0][:, np.newaxis] + pixel_corner_j_hat_coords[0]
        corner_coords_dec = center_dec + pixel_corner_i_hat_coords[1][:, np.newaxis] + pixel_corner_j_hat_coords[1]
        #input(f'center_ra, center_dec = {center_ra, center_dec}')
        #input(f'corner_coords_ra={corner_coords_ra}\n corner_coords_dec={corner_coords_dec}')

        def get_pixel_corner_coord(row, col, x, y):
            """Returns the RA-DEC coordinate of the corner (x,y) = (0..1, 0..1) of
            pixel (row, col) = (0..31, 0..31).
            Pixels and corner positions are zero-indexed from the top left."""
            #input(corner_coords_ra[col + y], corner_coords_dec[row + x])
            #coord_ra = center_ra + pixel_corner_i_hat_coords[0][col + y] + pixel_corner_j_hat_coords[0][row + x]
            #coord_dec = center_dec + pixel_corner_i_hat_coords[1][col + y] + pixel_corner_j_hat_coords[1][row + x]
            #return coord_ra, coord_dec
            return corner_coords_ra[row + x, col + y], corner_coords_dec[row + x, col + y]
            #print(f'i_hat * (col + y) = {i_hat * (col + y)}')
            #input(f'i_hat * (col + y) + j_hat * (row + x) = {i_hat * (col + y) + j_hat * (row + x)}')
            #input(f'col_offsets[col + y] = {col_offsets[col + y]}')
            #input(f'row_offsets[row + x] = {row_offsets[row + x]}')
            #coord = i_hat * col_offsets[col + y] + j_hat * row_offsets[row + x]
            #print(f'row, col, x, y = {row, col, x, y}, coord_ra = {center_ra + coord[0]}, coord_dec = {center_dec + coord[1]}')
            #print(center_ra + coord[0], center_dec + coord[1])
            #return center_ra + coord[0], center_dec + coord[1]
        return get_pixel_corner_coord
    return get_pixel_corner_coord_ftn


def get_module_corner_coords(center_ra, center_dec, pos_angle):
    """Return a 2x2 list containing the RA-DEC coordinates of the corners of a module's FoV."""
    get_pixel_coord = get_module_pixel_corner_coord_ftn(pos_angle)(center_ra, center_dec)
    corner_coords = []
    for i in range(2):
        row = []
        for j in range(2):
            coord = get_pixel_coord(i * 31, j * 31, i, j)
            row.append(coord)
        corner_coords.append(row)
    return corner_coords


def get_sky_band_corner_coords(t_start, t_end, azimuth, elevation, obslat, obslon, obsalt, pos_angle):
    """Return a 2x2 list of RA-DEC coordinates bounding the region of sky observed by a module
    between t_start and t_end. The positions are zero-indexed from the top left corner."""
    assert t_end >= t_start, 'End time cannot be before start time.'
    center_start_ra, center_start_dec = get_module_center_ra_dec(t_start, azimuth, elevation, obslat, obslon, obsalt)
    start_corner_coords = get_module_corner_coords(center_start_ra, center_start_dec, pos_angle)

    center_end_ra, center_end_dec = get_module_center_ra_dec(t_end, azimuth, elevation, obslat, obslon, obsalt)
    end_corner_coords = get_module_corner_coords(center_end_ra, center_end_dec, pos_angle)
    sky_band_corner_coords = [
        [start_corner_coords[0][0], end_corner_coords[0][1]],
        [start_corner_coords[1][0], end_corner_coords[1][1]]
    ]
    return sky_band_corner_coords


#print(get_sky_band_corner_coords(1665164683, 1665225883, 77, 77, 37.3414, 121.64292, 234))
