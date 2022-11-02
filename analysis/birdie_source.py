"""
BirdieSource objects are streams of synthetic signals used in birdie injection.
Each object is determined by a set of parameters, which specify birdie RA-DEC coordinates, pulse
creation times, durations, and intensities.

Objects can be initialized from a file containing pre-generated birdies or randomly generated. (TODO)
"""
import astropy.coordinates as c
import astropy.units as u
from birdie_utils import ra_dec_to_sky_array_indices
import numpy as np
import math

np.random.seed(10)


class BaseBirdieSource:
    """Base class for BirdieSource objects."""
    def __init__(self, birdie_config):
        self.config = birdie_config
        self.sky_coord = c.SkyCoord(
            ra=birdie_config['ra']*u.deg, dec=birdie_config['dec']*u.deg, frame='icrs'
        )
        self.max_dt = birdie_config['end_t'] - birdie_config['start_t']
        self.max_cycle_pos = birdie_config['duty_cycle'] * birdie_config['period']

    def __key(self):
        return self.config['ra'], self.config['dec'], self.config['start_t'], self.config['end_t'],\
               self.config['duty_cycle'], self.config['period'], self.config['intensity']

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, BaseBirdieSource):
            return self.__key() == other.__key()
        return NotImplemented

    def is_in_view(self, frame_utc):
        dt = frame_utc - self.config['start_t']
        if 0 <= dt <= self.max_dt:
            if math.fmod(dt, self.config['period']) <= self.max_cycle_pos:
                return True
        return False

    def generate_birdie(self, frame_utc, sky_array, bounding_box):
        """Generate a birdie and add it to sky_array. Returns a log entry for this pulse."""
        ax, ay = ra_dec_to_sky_array_indices(self.config['ra'], self.config['dec'], bounding_box)
        intensity = self.pulse_intensity(frame_utc)
        sky_array[ax, ay] = intensity
        return self.get_log_entry(intensity)

    def pulse_intensity(self, frame_utc):
        """Returns the intensity of this birdie at frame_utc in raw adc units."""
        return self.config['intensity']

    def get_log_entry(self, pulse_intensity):
        """A birdie log entry has the form:
            {
                'birdie_id': hash identifying a birdie source in birdie_sources.json,
                'intensity': intensity in raw adc of birdie
            }
        """
        log_entry = {
            'birdie_id': hash(self),
            'intensity': pulse_intensity
        }
        return log_entry
