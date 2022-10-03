

import astropy.coordinates as c
import astropy.units as u
import astropy.time as t


class ModuleView:
    """Simulates the FoV of a PanoSETI module on the celestial sphere."""
    # See pixel scale on https://oirlab.ucsd.edu/PANOinstr.html.
    pixels_per_side = 2

    def __init__(self, module_id, start_time_utc, obslat, obslon, obsalt, azimuth, elevation, pos_angle):
        self.module_id = module_id
        self.start_time_utc = start_time_utc
        self.azimuth = azimuth * u.deg
        self.elevation = elevation * u.deg
        self.pos_angle = pos_angle
        self.earth_loc = c.EarthLocation(
            lat=obslat*u.deg, lon=obslon*u.deg, height=obsalt*u.m, ellipsoid='WGS84'
        )
        self.current_pos = self.initial_pos = self.get_center_ra_dec_coords(start_time_utc)
        self.pixel_array = []
        for i in range(self.pixels_per_side):
            row = []
            for j in range(self.pixels_per_side):
                row.append(PixelView(self, i, j))
            self.pixel_array.append(row)

    def get_center_ra_dec_coords(self, frame_utc):
        """Return the RA-DEC coordinates of the center of the module's field of view at frame_utc."""
        time = t.Time(frame_utc, format='unix')
        alt_alz_coords = c.SkyCoord(
            az=self.azimuth, alt=self.elevation, location=self.earth_loc,
            obstime=time, frame=c.AltAz
        )
        return alt_alz_coords.transform_to('icrs')


class PixelView:
    """Simulates the FoV of a single pixel on the celestial sphere."""
    pixel_scale = 0.31

    def __init__(self, module_view_obj: ModuleView, px, py):
        assert 0 <= px <= 31 and 0 <= py <= 31
        self.module = module_view_obj
        self.px = px
        self.py = py
        # RA-DEC coordinate offsets from the module's current position
        # to this pixel's bottom left corner.
        self.ra_offset = c.Angle((px - ModuleView.pixels_per_side // 2) * self.pixel_scale, unit='deg')
        self.dec_offset = c.Angle((py - ModuleView.pixels_per_side // 2) * self.pixel_scale, unit='deg')
        self.current_pos = None
        self.update_pos()

    def update_pos(self):
        self.current_pos = self.module.current_pos.spherical_offsets_by(self.ra_offset, self.dec_offset)

    def __repr__(self):
        return f'Pixel({self.px}, {self.py}) at {self.current_pos}'


m1 = ModuleView(42, 1656443180, 10.3, 44.2, 234, 77, 77, 77)
print(m1.initial_pos)
for i in m1.pixel_array:
    for j in i:
        print(j)

p0_0 = PixelView(m1, 0, 0)
print(p0_0.ra_offset)
print(p0_0.dec_offset)
print(p0_0.current_pos)

"""
m2 = ModuleView(42, 1664776735, 10.3, 44.2, 234, 77, 77, 77)
print(m2.initial_pos)
m3 = ModuleView(42, 1664782234, 10.3, 44.2, 234, 77, 77, 77)
print(m3.initial_pos)
print(m3.current_pos)
"""