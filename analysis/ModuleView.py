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
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate

import birdie_injection_utils as utils
import time


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
        self.ra_offsets = self.dec_offsets = None
        self.init_offset_arrays()
        self.pixel_grid_ra = self.pixel_grid_dec = None
        self.current_pos = None
        self.update_center_ra_dec_coords(start_time_utc)
        # Simulated data
        self.simulated_img_arr = np.zeros(self.pixels_per_side**2, dtype=np.int16)
        self.sky_band = None

    def init_offset_arrays(self):
        shape = (self.pixels_per_side + 1, self.pixels_per_side + 1)
        self.ra_offsets = np.empty(shape)
        self.dec_offsets = np.empty(shape)
        # Populate ra_offsets and dec_offsets.
        for i in range(self.pixels_per_side + 1):
            ra_offset = (i - self.pixels_per_side // 2) * self.pixel_scale
            for j in range(self.pixels_per_side + 1):
                dec_offset = (-j + self.pixels_per_side // 2) * self.pixel_scale
                self.ra_offsets[i, j] = ra_offset
                self.dec_offsets[i, j] = dec_offset

    def set_pixel_value(self, px, py, val):
        """In the simulated image frame, set the value of pixel (px, py) to val."""
        if val < 0:
            val = 0
        elif val > self.max_pixel_counter_value:
            val = self.max_pixel_counter_value
        index_1d = px * self.pixels_per_side + py
        self.simulated_img_arr[index_1d] = val

    def clear_simulated_img_arr(self):
        self.simulated_img_arr.fill(0)

    def plot_simulated_image(self):
        """Converts the 1x1024 simulated img array to a 32x32 array."""
        s = self.pixels_per_side
        img = np.zeros((s, s,))
        for row in range(s):
            img[row] = self.simulated_img_arr[s*row:s*(row+1)]
        fig1, ax = plt.subplots()
        ax.pcolormesh(np.arange(s), np.arange(s), img, vmin=0, vmax=150)
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        return img

    def update_pixel_coord_grid(self):
        """Updates the coordinates associated with each pixel corner in this module."""
        current_ra = self.current_pos.ra.value
        current_dec = self.current_pos.dec.value
        self.pixel_grid_ra = self.ra_offsets + current_ra
        self.pixel_grid_dec = self.dec_offsets + current_dec
        #print(f'ra grid = {self.pixel_grid_ra}\ndec grid = {self.pixel_grid_dec}')

    def update_center_ra_dec_coords(self, frame_utc):
        """Return the RA-DEC coordinates of the center of the module's field of view at frame_utc."""
        obstime = t.Time(frame_utc, format='unix')
        alt_alz_coords = c.SkyCoord(
            az=self.azimuth, alt=self.elevation, location=self.earth_loc,
            obstime=obstime, frame=c.AltAz
        )
        self.current_pos = alt_alz_coords.transform_to('icrs')
        self.update_pixel_coord_grid()

    def simulate_one_pixel_fov(self, px, py, sky_array):
        """Sum the intensities in each element of sky_array visible by pixel (px, py) and return a counter value
        to add to the current image frame. We approximate the pixel FoV as a square determined by the RA-DEC
         coordinates of the top left and bottom right corner of the pixel."""
        total_intensity = 0.0
        left_index, high_index = utils.ra_dec_to_sky_array_indices(
            self.pixel_grid_ra[px, py], self.pixel_grid_dec[px, py], sky_array)
        right_index, low_index = utils.ra_dec_to_sky_array_indices(
            self.pixel_grid_ra[px + 1, py + 1], self.pixel_grid_dec[px + 1, py + 1], sky_array)
        #print(f'(x0, y0) = ({x0}, {y0}). (x1, y1) = ({x1}, {y1})')
        max_ra_index = sky_array.shape[0]
        # RA coordinates may wrap around.
        if left_index > right_index:
            self.sky_band[left_index:, low_index:high_index + 1] = 50
            self.sky_band[:right_index + 1, low_index:high_index + 1] = 50
            l_sum = sky_array[left_index:, low_index:high_index + 1].sum()
            r_sum = sky_array[:right_index + 1, low_index:high_index + 1].sum()
            total_intensity += l_sum + r_sum
        else:
            self.sky_band[left_index:right_index + 1, low_index:high_index + 1] = 50
            total_intensity += sky_array[left_index:right_index + 1, low_index:high_index + 1].sum()
        self.set_pixel_value(px, py, round(total_intensity))

    def simulate_all_pixel_fovs(self, sky_array):
        if self.sky_band is None:
            self.sky_band = np.zeros(sky_array.shape)
        for i in range(self.pixels_per_side):
            for j in range(self.pixels_per_side):
                self.simulate_one_pixel_fov(i, j, sky_array)
