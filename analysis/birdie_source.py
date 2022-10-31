"""
BirdieSource objects generate collections of signals for the signal injection and recovery program,
which we call 'birdie injection' after a similarly named practice in radio astronomy.
Each object is determined by a set of parameters, which specify birdie RA-DEC coordinates and pulse
creation times, durations, and intensities.

Objects can be initialized from a file containing pre-generated birdies or randomly generated.
"""
import astropy.coordinates as c
import astropy.units as u
from birdie_utils import ra_dec_to_sky_array_indices
import numpy as np
import math
np.random.seed(10)


class BaseBirdieSource:
    """Base class for BirdieSource objects."""
    def __init__(self,
                 ra,
                 dec,
                 start_utc,
                 end_utc,
                 duty_cycle,
                 period,
                 intensity):
        self.ra = ra % 360
        self.dec = dec
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.duty_cycle=duty_cycle
        self.period = period
        self.intensity = intensity
        self.sky_coord = c.SkyCoord(
            ra=ra*u.deg, dec=dec*u.deg, frame='icrs'
        )
        self.max_dt = self.end_utc - self.start_utc
        self.max_cycle_pos = self.duty_cycle * self.period

    def is_in_view(self, frame_utc):
        dt = frame_utc - self.start_utc
        if 0 <= dt <= self.max_dt:
            if math.fmod(dt, self.period) <= self.max_cycle_pos:
                return True
        return False

    def generate_birdie(self, frame_utc, sky_array, bounding_box):
        """Generate a birdie and add it to sky_array. Returns True if sky_array is modified and False otherwise."""
        ax, ay = ra_dec_to_sky_array_indices(self.ra, self.dec, bounding_box)
        sky_array[ax, ay] = self.pulse_intensity(frame_utc)

    def pulse_intensity(self, frame_utc):
        """Returns the intensity of this birdie at frame_utc in raw adc units."""
        return self.intensity

    def get_log_entry(self):
        pass
