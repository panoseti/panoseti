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

from birdie_utils import ra_dec_to_sky_array_indices, bresenham_line, get_coord_bounding_box
from sky_band import get_module_pixel_corner_coord_ftn


class ModuleView:
    """Simulates the FoV of a PanoSETI module on the celestial sphere."""
    # See pixel size on https://oirlab.ucsd.edu/PANOinstr.html.
    pixels_per_side = 32
    pixel_size = 0.31
    max_pixel_counter_value = 2**8 - 1

    def __init__(self, module_id, start_time_utc, obslat, obslon, obsalt, azimuth, elevation, pos_angle, bytes_per_pixel, sky_array):
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
        self.pixel_fovs = dict()
        self.init_center_ra_dec_coords(start_time_utc, sky_array)
        # Simulated data array. dtype is double the max possible value
        if bytes_per_pixel == 1:
            self.simulated_img_arr = np.zeros(self.pixels_per_side**2, dtype=np.uint16)
            self.max_pixel_counter_value = 2**8 - 1
        elif bytes_per_pixel == 2:
            self.simulated_img_arr = np.zeros(self.pixels_per_side**2, dtype=np.uint32)
            self.max_pixel_counter_value = 2**16 - 1
        else:
            raise Warning(f'bytes_per_pixel must be 1 or 2, not "{bytes_per_pixel}"')

    def __str__(self):
        loc_geodetic = self.earth_loc.to_geodetic(ellipsoid="WGS84")
        s = f'Module "{self.module_id}" view @ {datetime.datetime.fromtimestamp(self.current_utc)}\n' \
            f'centered at RA={round(self.center_ra, 1):>6}<deg>, DEC={round(self.center_dec, 1):5>}<deg>, pos angle={self.pos_angle}<deg>'
        """
        f'Lon={round(loc_geodetic.lon.value, 3)}<deg>, ' \
        f'Lat={round(loc_geodetic.lat.value, 3)}<deg>, ' \
        f'Height={round(loc_geodetic.height.value, 3)}<m>'
        """
        return s

    def init_center_ra_dec_coords(self, frame_utc, sky_array):
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
        self.init_pixel_rasters(sky_array)

    def init_pixel_rasters(self, sky_array):
        """Initialize self.pixel_fovs, a dictionary of pixel coordinate: pixel raster pairs.
        Each pixel raster is a view into sky_array corresponding to the region of the sky
        visible by that detector."""
        bounding_box = get_coord_bounding_box(self.center_ra, self.center_dec)
        for px in range(32):
            for py in range(32):
                self.pixel_fovs[px, py] = self.compute_pixel_raster(px, py, bounding_box, sky_array)

    def compute_pixel_raster(self, px, py, bounding_box, sky_array):
        """Return a masked numpy view into sky_array corresponding to the scanline rasterization
        of the region of sky_array visible by the detector at coordinates px, py.
        The program uses top left corner zero-indexing."""
        # Get sky_array indices for the corners of detector (px, py)
        indices = [-1] * 4
        for row in range(2):
            for col in range(2):
                x, y = self.get_pixel_corner_coord(px, py, row, col)
                indices[2 * row + col] = ra_dec_to_sky_array_indices(x, y, bounding_box)
        # Get pixel horizontal slices part of this pixel's FoV.
        min_x, max_x = min(indices, key=lambda p: p[0])[0], max(indices, key=lambda p: p[0])[0]
        min_y, max_y = min(indices, key=lambda p: p[1])[1], max(indices, key=lambda p: p[1])[1]
        # Dictionary of [y-index] : [min_x index, max_x index].
        pts = {
            y: [float('inf'), float('-inf')] for y in range(min_y, max_y + 1)
        }
        corners = [0, 1, 3, 2, 0]
        for i in range(4):
            x1, y1 = indices[corners[i]]
            x0, y0 = indices[corners[i + 1]]
            bresenham_line(x0, y0, x1, y1, pts)
        # Create masked numpy view:
        mask_shape = max_y - min_y + 1, max_x - min_x + 1
        pixel_mask = np.ones(mask_shape)
        for y in pts:
            left, right = pts[y][0] - min_x, pts[y][1] - min_x + 1
            pixel_mask[y - min_y, left:right] = 0
        # Get a view into the portion of sky_array corresponding to the FoV of this pixel.
        pixel_sky_array_slice = sky_array[min_y:max_y + 1, min_x:max_x + 1]
        return np.ma.array(pixel_sky_array_slice, mask=pixel_mask)

    def update_center_ra_dec_coords(self, frame_utc):
        """Return the RA-DEC coordinates of the center of the module's field of view at frame_utc."""
        assert frame_utc >= self.current_utc, f'frame_utc must be at least as large as self.current_utc'
        dt = frame_utc - self.current_utc
        # Earth rotates approx 360 degrees in 24 hrs.)
        self.center_ra = (self.center_ra + dt * 360 / (24 * 60 * 60)) % 360
        self.current_utc = frame_utc
        self.get_pixel_corner_coord = self.get_module_pixel_corner_coord_ftn(self.center_ra, self.center_dec)
        #print(f'center_ra, center_dec = {self.center_ra, self.center_dec}')

    def add_birdies_to_image_array(self, raw_img):
        assert len(raw_img) == len(self.simulated_img_arr)
        raw_with_birdies = raw_img + self.simulated_img_arr
        return np.clip(raw_with_birdies, 0, self.max_pixel_counter_value)
        #indices_above_max_counter_val = np.nonzero(raw_with_birdies > self.max_pixel_counter_value)
        #raw_with_birdies[indices_above_max_counter_val] = self.max_pixel_counter_value

    def simulate_all_pixel_fovs(self):
        """Simulate every pixel FoV in this module, resulting in a simulated 32x32 image array
        containing only birdies."""
        for px in range(32):
            for py in range(32):
                # Sum the intensities in each element of sky_array visible by pixel (px, py) and
                # return a counter value to add to the current image frame.
                total_intensity = self.pixel_fovs[px, py].sum()
                self.simulated_img_arr[px * 32 + py] = min(total_intensity, self.max_pixel_counter_value)

    def plot_simulated_image(self, raw_img):
        """Plot the simulated image array."""
        s = self.pixels_per_side
        raw_with_birdies = self.add_birdies_to_image_array(raw_img)
        raw_with_birdies_32x32 = np.resize(raw_with_birdies, (32, 32))
        fig1, ax = plt.subplots()
        ax.pcolormesh(np.arange(s), np.arange(s), raw_with_birdies_32x32, vmin=0, vmax=3000)
        ax.set_aspect('equal', adjustable='box')
        fig1.suptitle(self)
        #fig1.show()
        return fig1
