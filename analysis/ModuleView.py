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
import datetime
import math

import astropy.coordinates as c
import astropy.units as u
import astropy.time as t
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

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
        self.azimuth = azimuth * u.deg
        self.elevation = elevation * u.deg
        self.pos_angle = pos_angle
        self.earth_loc = c.EarthLocation(
            lat=obslat*u.deg, lon=obslon*u.deg, height=obsalt*u.m, ellipsoid='WGS84'
        )
        # Current field of view RA-DEC coordinates
        self.current_utc = start_time_utc
        self.center_ra = self.center_dec = None
        self.ra_offsets = self.dec_offsets = None
        self.pixel_grid_ra = self.pixel_grid_dec = None
        self.init_offset_arrays()
        self.update_center_ra_dec_coords(start_time_utc)
        # Simulated data
        self.simulated_img_arr = np.zeros(self.pixels_per_side**2, dtype=np.int16)
        self.sky_band = None

    def __str__(self):
        loc_geodetic = self.earth_loc.to_geodetic(ellipsoid="WGS84")
        s = f'Module "{self.module_id}" view @ {datetime.datetime.fromtimestamp(self.current_utc)}\n' \
            f'centered at RA={round(self.center_ra, 1):>6}<deg>, DEC={round(self.center_dec, 1):5>}<deg>'
        """
        f'Lon={round(loc_geodetic.lon.value, 3)}<deg>, ' \
        f'Lat={round(loc_geodetic.lat.value, 3)}<deg>, ' \
        f'Height={round(loc_geodetic.height.value, 3)}<m>'
        """
        return s

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
        # FAST
        if val < 0:
            val = 0
        elif val > self.max_pixel_counter_value:
            val = self.max_pixel_counter_value
        index_1d = px * self.pixels_per_side + py
        self.simulated_img_arr[index_1d] = val

    def clear_simulated_img_arr(self):
        self.simulated_img_arr.fill(0)

    def add_birdies_to_image_array(self, raw_img):
        assert len(raw_img) == len(self.simulated_img_arr)
        raw_with_birdies = raw_img+ self.simulated_img_arr
        indices_above_max_counter_val = np.nonzero(raw_with_birdies > self.max_pixel_counter_value)
        raw_with_birdies[indices_above_max_counter_val] = self.max_pixel_counter_value
        return raw_with_birdies

    def plot_simulated_image(self, raw_img):
        """Plot the simulated image array."""
        s = self.pixels_per_side
        raw_with_birdies = self.add_birdies_to_image_array(raw_img)
        raw_with_birdies_32x32 = np.resize(raw_with_birdies, (32, 32))
        fig1, ax = plt.subplots()
        ax.pcolormesh(np.arange(s), np.arange(s), raw_with_birdies_32x32, vmin=0, vmax=255)
        ax.set_aspect('equal', adjustable='box')
        fig1.suptitle(self)
        fig1.show()

    def plot_sky_band(self):
        """Plot a heatmap of the sky covered during the simulation."""
        utils.graph_sky_array(self.sky_band)

    def update_pixel_coord_grid(self):
        """Updates the coordinates associated with each pixel corner in this module."""
        # FAST
        self.pixel_grid_ra = self.ra_offsets + self.center_ra
        self.pixel_grid_dec = self.dec_offsets + self.center_dec
        #print(f'ra grid = {self.pixel_grid_ra}\ndec grid = {self.pixel_grid_dec}')

    def get_module_ra_at_time(self, frame_utc):
        """Return the new """
        # Earth rotates approx 360 degrees in 24 hrs.
        dt = frame_utc - self.current_utc
        return self.center_ra + dt * 360 / (24 * 60 * 60)

    def update_center_ra_dec_coords(self, frame_utc):
        """Return the RA-DEC coordinates of the center of the module's field of view at frame_utc."""
        if self.center_ra is None or self.center_dec is None:
            obstime = t.Time(frame_utc, format='unix')
            alt_alz_coords = c.SkyCoord(
                az=self.azimuth, alt=self.elevation, location=self.earth_loc,
                obstime=obstime, frame=c.AltAz
            )
            pos = alt_alz_coords.transform_to('icrs')
            self.center_ra = pos.ra.value
            self.center_dec = pos.dec.value
        else:
            assert frame_utc >= self.current_utc, f'frame_utc must be at least as large as self.current_utc'
            self.center_ra = self.get_module_ra_at_time(frame_utc) % 360
            self.current_utc = frame_utc
        self.update_pixel_coord_grid()

    def simulate_one_pixel_fov(self, px, py, sky_array, draw_sky_band):
        """Sum the intensities in each element of sky_array visible by pixel (px, py) and return a counter value
        to add to the current image frame. We approximate the pixel FoV as a square determined by the RA-DEC
         coordinates of the top left and bottom right corner of the pixel."""
        total_intensity = 0.0
        left_index, high_index = utils.ra_dec_to_sky_array_indices(
            self.pixel_grid_ra[px, py], self.pixel_grid_dec[px, py], sky_array
        )
        right_index, low_index = utils.ra_dec_to_sky_array_indices(
            self.pixel_grid_ra[px + 1, py + 1], self.pixel_grid_dec[px + 1, py + 1], sky_array
        )
        max_ra_index = sky_array.shape[0]
        # RA coordinates may wrap around if larger than 24hrs.
        if left_index > right_index:
            if draw_sky_band:
                self.sky_band[left_index:, low_index:high_index + 1] += 5
                self.sky_band[:right_index + 1, low_index:high_index + 1] += 5
            left = sky_array[left_index:, low_index:high_index + 1]
            right = sky_array[:right_index + 1, low_index:high_index + 1]
            total_intensity += (left + right).sum()
        else:
            if draw_sky_band:
                self.sky_band[left_index:right_index + 1, low_index:high_index + 1] += 5
            total_intensity += sky_array[left_index:right_index + 1, low_index:high_index + 1].sum()
        self.set_pixel_value(px, py, math.floor(total_intensity))

    def simulate_all_pixel_fovs(self, sky_array, draw_sky_band=False):
        """Simulate every pixel FoV in this module, resulting in a simulated 32x32 image array
        containing only birdies."""
        if self.sky_band is None:
            self.sky_band = np.copy(sky_array)
        for i in range(self.pixels_per_side):
            for j in range(self.pixels_per_side):
                self.simulate_one_pixel_fov(i, j, sky_array, draw_sky_band)
