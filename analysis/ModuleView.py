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

from birdie_injection_utils import ra_dec_to_sky_array_indices, graph_sky_array, line
import sky_band
from sky_band import get_module_pixel_corner_coord_ftn


class ModuleView:
    """Simulates the FoV of a PanoSETI module on the celestial sphere."""
    # See pixel scale on https://oirlab.ucsd.edu/PANOinstr.html.
    pixels_per_side = 32
    pixel_size = 0.31
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
        self.get_pixel_corner_coord = None
        self.get_module_pixel_corner_coord_ftn = get_module_pixel_corner_coord_ftn(
            pos_angle, pixel_size=ModuleView.pixel_size
        )
        self.init_center_ra_dec_coords(start_time_utc)
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

    def init_center_ra_dec_coords(self, frame_utc):
        if self.center_ra is None or self.center_dec is None:
            obstime = t.Time(frame_utc, format='unix')
            alt_alz_coords = c.SkyCoord(
                az=self.azimuth, alt=self.elevation, location=self.earth_loc,
                obstime=obstime, frame=c.AltAz
            )
            pos = alt_alz_coords.transform_to('icrs')
            self.center_ra = pos.ra.value
            self.center_dec = pos.dec.value
        self.get_pixel_corner_coord = self.get_module_pixel_corner_coord_ftn(self.center_ra, self.center_dec)

    def add_birdies_to_image_array(self, raw_img):
        assert len(raw_img) == len(self.simulated_img_arr)
        raw_with_birdies = raw_img + self.simulated_img_arr
        #indices_above_max_counter_val = np.nonzero(raw_with_birdies > self.max_pixel_counter_value)
        #raw_with_birdies[indices_above_max_counter_val] = self.max_pixel_counter_value
        return raw_with_birdies

    def plot_simulated_image(self, raw_img):
        """Plot the simulated image array."""
        s = self.pixels_per_side
        raw_with_birdies = self.add_birdies_to_image_array(raw_img)
        raw_with_birdies_32x32 = np.resize(raw_with_birdies, (32, 32))
        fig1, ax = plt.subplots()
        ax.pcolormesh(np.arange(s), np.arange(s), raw_with_birdies_32x32, vmin=0, vmax=2000)
        ax.set_aspect('equal', adjustable='box')
        fig1.suptitle(self)
        fig1.show()

    def plot_sky_band(self):
        """Plot a heatmap of the sky covered during the simulation."""
        graph_sky_array(self.sky_band)

    def update_center_ra_dec_coords(self, frame_utc):
        """Return the RA-DEC coordinates of the center of the module's field of view at frame_utc."""
        assert frame_utc >= self.current_utc, f'frame_utc must be at least as large as self.current_utc'
        dt = frame_utc - self.current_utc
        # Earth rotates approx 360 degrees in 24 hrs.)
        self.center_ra = (self.center_ra + dt * 360 / (24 * 60 * 60)) % 360
        self.current_utc = frame_utc
        self.get_pixel_corner_coord = self.get_module_pixel_corner_coord_ftn(self.center_ra, self.center_dec)
        print(f'center_ra, center_dec = {self.center_ra, self.center_dec}')

    def simulate_one_pixel_fov(self, px, py, sky_array, bounding_box, draw_sky_band):
        """Sum the intensities in each element of sky_array visible by pixel (px, py) and return a counter value
        to add to the current image frame. We approximate the pixel FoV as a square determined by the RA-DEC
         coordinates of the top left and bottom right corner of the pixel."""
        left_ra, high_dec = self.get_pixel_corner_coord(px, py, 0, 0)
        right_ra, low_dec = self.get_pixel_corner_coord(px, py, 1, 1)

        left_index, high_index = ra_dec_to_sky_array_indices(left_ra, high_dec, bounding_box)
        right_index, low_index = ra_dec_to_sky_array_indices(right_ra, low_dec, bounding_box)

        if draw_sky_band:
            self.sky_band[left_index:right_index + 1, low_index:high_index + 1] += 5
        total_intensity = math.floor(sky_array[left_index:right_index + 1, low_index:high_index + 1].sum())
        # Set pixel value
        self.pixel_convex_raster(px, py, bounding_box)
        if total_intensity > self.max_pixel_counter_value:
            total_intensity = self.max_pixel_counter_value
        self.simulated_img_arr[px * self.pixels_per_side + py] = total_intensity

    def pixel_convex_raster(self, px, py, bounding_box):
        # Get sky_array indices for the corners of detector (px, py)
        indices = [-1] * 4
        for row in range(2):
            for col in range(2):
                x, y = self.get_pixel_corner_coord(px, py, row, col)
                indices[2*row + col] = ra_dec_to_sky_array_indices(x, y, bounding_box)
        #min_x, max_x = min(indices, key=lambda p: p[0]), max(indices, key=lambda p: p[0])
        min_y, max_y = min(indices, key=lambda p: p[1])[1], max(indices, key=lambda p: p[1])[1]
        #print(indices)
        pts = {y: [float('inf'), float('-inf')] for y in range(min_y, max_y + 1)}
        corners = [0, 1, 3, 2, 0]
        for i in range(4):
            x1, y1 = indices[corners[i]]
            x0, y0 = indices[corners[i + 1]]
            line(x0, y0, x1, y1, pts)

        #print(f'pts={pts}')
        for y in pts:
            for x in range(pts[y][0], pts[y][1] + 1):
                self.sky_band[x, y] = 10000

    def simulate_all_pixel_fovs(self, sky_array, bounding_box, draw_sky_band=False):
        """Simulate every pixel FoV in this module, resulting in a simulated 32x32 image array
        containing only birdies."""
        print(bounding_box)
        if self.sky_band is None:
            self.sky_band = np.copy(sky_array)
        for i in range(self.pixels_per_side):
            for j in range(self.pixels_per_side):
                self.simulate_one_pixel_fov(i, j, sky_array, bounding_box, draw_sky_band)
        self.plot_sky_band()
        input(f'i, j = {i, j}')
