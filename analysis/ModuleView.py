"""
A ModuleView object simulates the on-sky field of view (FoV) of a PanoSETI module during an observing run.
The FoV is determined by
    1. the fixed Alt-AZ orientation of the module specified in obs_config.json
    2. a utc timestamp, determining the RA-DEC coordinates in view.
Given a utc timestamp, the RA-DEC coordinates of the center of the module's FoV are calculated with the ICRS system.
Then, the coordinates of each pixel are computed relative to the module's center.
These coordinates are used to determine the FoV of each pixel and integrate over the visible simulated signals
 produced by BirdieSource objects.
"""

import astropy.coordinates as c
import astropy.units as u
import astropy.time as t
import numpy as np

import birdie_injection_utils as utils


class ModuleView:
    """Simulates the FoV of a PanoSETI module on the celestial sphere."""
    # See pixel scale on https://oirlab.ucsd.edu/PANOinstr.html.
    pixels_per_side = 32
    pixel_scale = 0.31
    max_pixel_counter_value = 2**8 - 1

    def __init__(self, module_id, start_time_utc, obslat, obslon, obsalt, azimuth, elevation, pos_angle):
        self.module_id = module_id
        # Module orientation
        self.start_time_utc = start_time_utc
        self.azimuth = azimuth * u.deg
        self.elevation = elevation * u.deg
        self.pos_angle = pos_angle
        self.earth_loc = c.EarthLocation(
            lat=obslat*u.deg, lon=obslon*u.deg, height=obsalt*u.m, ellipsoid='WGS84'
        )
        # Current field of view RA-DEC coordinates
        self.current_pixel_corner_coords = None
        self.current_pos = None
        self.update_center_ra_dec_coords(start_time_utc)
        # Simulated data
        self.simulated_img_arr = np.zeros(self.pixels_per_side**2)

    def set_pixel_value(self, px, py, val):
        """In the simulated image frame, set the value of pixel (px, py) to val."""
        if val < 0:
            val = 0
        elif val > self.max_pixel_counter_value:
            val = self.max_pixel_counter_value
        self.simulated_img_arr[px, py] = val

    def clear_simulated_img_arr(self):
        self.simulated_img_arr.fill(0)

    def update_center_ra_dec_coords(self, frame_utc):
        """Return the RA-DEC coordinates of the center of the module's field of view at frame_utc."""
        time = t.Time(frame_utc, format='unix')
        alt_alz_coords = c.SkyCoord(
            az=self.azimuth, alt=self.elevation, location=self.earth_loc,
            obstime=time, frame=c.AltAz
        )
        self.current_pos = alt_alz_coords.transform_to('icrs')
        self.update_pixel_coord_grid()

    def update_pixel_coord_grid(self):
        """Updates the coordinates associated with each pixel corner in this module."""
        ra_offsets = []
        dec_offsets = []
        # Populate ra_offsets and dec_offsets.
        for i in range(self.pixels_per_side + 1):
            ra_offset = (i - self.pixels_per_side // 2) * self.pixel_scale
            for j in range(self.pixels_per_side + 1):
                dec_offset = (-j + self.pixels_per_side // 2) * self.pixel_scale
                ra_offsets.append(ra_offset)
                dec_offsets.append(dec_offset)
        #print(f'ra_offsets = {ra_offsets}\ndec_offsets = {dec_offsets}')
        self.current_pixel_corner_coords = self.current_pos.spherical_offsets_by(
            ra_offsets * u.deg, dec_offsets * u.deg
        )

    def get_pixel_coord(self, i, j):
        """Return a tuple containing the RA-DEC coordinates of the pixel corner at index (i,j)."""
        index_1d = i * (self.pixels_per_side + 1) + j
        pixel_sky_coord = self.current_pixel_corner_coords[index_1d]
        pixel_ra = pixel_sky_coord.ra / u.deg
        pixel_dec = pixel_sky_coord.dec / u.deg
        return pixel_ra, pixel_dec

    def get_simulated_pixel_data(self, px, py, sky_array):
        """Sum the intensities in each element of sky_array visible by pixel (px, py) and return a counter value
        to add to the current image frame."""
        # Coords of bottom left corner of the pixel:
        x0, y0 = utils.ra_dec_to_sky_array_indices(*self.get_pixel_coord(px, py), sky_array)
        print(x0, y0)


num_ra = 360
arr = utils.get_sky_image_array(num_ra)

m1 = ModuleView(42, 1656443180, 10.3, 44.2, 234, 77, 77, 77)


m1.get_simulated_pixel_data(0, 0, arr)

m1.update_center_ra_dec_coords(1657443180)
m1.get_simulated_pixel_data(0, 0, arr)

# Wraps around...
m1.update_center_ra_dec_coords(1694810080)
m1.get_simulated_pixel_data(0, 0, arr)


for i in m1.current_pixel_corner_coords:
    print(i)
"""
m2 = ModuleView(42, 1664776735, 10.3, 44.2, 234, 77, 77, 77)
print(m2.initial_pos)
m3 = ModuleView(42, 1664782234, 10.3, 44.2, 234, 77, 77, 77)
print(m3.initial_pos)
print(m3.current_pos)
"""
