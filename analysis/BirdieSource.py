"""
BirdieSource objects generate collections of signals for the signal injection and recovery program,
which we call 'birdie injection' after a similarly named practice in radio astronomy.
Each object is determined by a set of parameters, which specify birdie RA-DEC coordinates and pulse
creation times, durations, and intensities.

Objects can be initialized from a file containing pre-generated birdies or randomly generated.
"""
from astropy import coordinates

class BaseBirdieSource:
    """Base class for BirdieSource objects."""

    def __init__(self, ra, dec, pulse_length=None, period=None):
        icrs = coordinates.ICRS(ra=ra, dec=dec)
        self.sky_coord = coordinates.SkyCoord(frame=icrs, unit='deg')



b = BaseBirdieSource(5, 5)
