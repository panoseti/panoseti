

import astropy.coordinates as c
import astropy.units as u
import astropy.time as t

class ModuleView:
    """Simulate the FoV of a PanoSETI module on the celestial sphere."""
    def __init__(self, module_id, start_time, obslat, obslon, obsalt, azimuth, elevation, pos_angle):
        self.module_id = module_id
        self.start_time = t.Time(start_time)
        self.ra_dec_initial = self.get_ra_dec_initial(obslat, obslon, obsalt, azimuth, elevation)
        self.pos_angle = pos_angle

    def get_ra_dec_initial(self, obslat, obslon, obsalt, azimuth, elevation):
        earth_loc = c.EarthLocation(
            lat=obslat*u.deg, lon=obslon*u.deg, height=obsalt*u.m, ellipsoid='WGS84'
        )
        alt_alz_coords = c.SkyCoord(
            az=azimuth*u.deg, alt=elevation*u.deg, location=earth_loc,
            obstime=self.start_time, frame=c.AltAz
        )
        return alt_alz_coords.transform_to('icrs')



m = ModuleView(100, '2022-07-30 23:00:00', 10.3, 44.2, 234, 77, 77, 77)
print(m.ra_dec_initial)