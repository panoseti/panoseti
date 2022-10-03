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
    def __init__(self, ra, dec, start_date=None, end_date=None, pulse_duration=None, period=None, intensity=150):
        #assert 0 <= ra <= 24 and -90 <= dec <= 90
        self.start_date = start_date
        self.end_date = end_date
        self.pulse_duration = pulse_duration
        self.period = period
        self.intensity = intensity
        self.ra = ra
        self.dec = dec
        self.sky_coord = c.SkyCoord(
            ra=ra*u.deg, dec=dec*u.deg, frame='icrs'
        )

    def generate_birdie(self, img_array, frame_utc):
        """Generate a birdie and add it to img_array."""
        arrx, arry = utils.ra_dec_to_img_array_indices(self.ra, self.dec, img_array)
        print(f'arrx = {arrx}, arry = {arry}')
        img_array[arrx, arry] = self.pulse_intensity(frame_utc)

    def pulse_intensity(self, frame_utc):
        """Returns the intensity of this birdie at frame_utc in raw adc units."""
        return self.intensity
