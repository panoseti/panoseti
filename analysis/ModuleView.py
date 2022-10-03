

import astropy.coordinates as c
import astropy.units as u
import astropy.time as t

class ModuleView:
    """Simulate the FoV of a PanoSETI module on the celestial sphere."""
    def __init__(self, module_id, start_time_utc, obslat, obslon, obsalt, azimuth, elevation, pos_angle):
        self.module_id = module_id
        self.start_time_utc = start_time_utc
        self.azimuth = azimuth * u.deg
        self.elevation = elevation * u.deg
        self.earth_loc = c.EarthLocation(
            lat=obslat*u.deg, lon=obslon*u.deg, height=obsalt*u.m, ellipsoid='WGS84'
        )
        self.current_pos = self.initial_pos = self.get_fov_ra_dec_coords(start_time_utc)
        self.pos_angle = pos_angle

    def get_fov_ra_dec_coords(self, frame_utc):
        """Return the RA-DEC coordinates of the center of the module's field of view at frame_utc."""
        time = t.Time(frame_utc, format='unix')
        alt_alz_coords = c.SkyCoord(
            az=self.azimuth, alt=self.elevation, location=self.earth_loc,
            obstime=time, frame=c.AltAz
        )
        return alt_alz_coords.transform_to('icrs')


m1 = ModuleView(42, 1656443180, 10.3, 44.2, 234, 77, 77, 77)
print(m1.initial_pos)
m2 = ModuleView(42, 1664776735, 10.3, 44.2, 234, 77, 77, 77)
print(m2.initial_pos)
m3 = ModuleView(42, 1664782234, 10.3, 44.2, 234, 77, 77, 77)
print(m3.initial_pos)
print(m3.current_pos)
