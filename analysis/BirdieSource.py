"""
BirdieSource objects generate collections of signals for the signal injection and recovery program,
which we call 'birdie injection' after a similarly named practice in radio astronomy.
Each object is determined by a set of parameters, which specify birdie RA-DEC coordinates and pulse
creation times, durations, and intensities.

Objects can be initialized from a file containing pre-generated birdies or randomly generated.
"""
import astropy.coordinates as c
import astropy.units as u
import birdie_injection_utils as utils


class BaseBirdieSource:
    """Base class for BirdieSource objects."""
    def __init__(self, ra, dec, start_date=None, pulse_length=None, period=None):
        assert 0 <= ra <= 24 and -90 <= dec <= 90
        self.sky_coord = c.SkyCoord(
            ra=ra*u.degree, dec=dec*u.degree, frame='icrs'
        )

    def generate_point(self, img_array):
        pass


b = BaseBirdieSource(5, 5)
