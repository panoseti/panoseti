function [Alt Az] = RADec2AltAz(RA,Dec,Latitude,Longitude,time)
% By: Christopher Wilcox and Ty Martinez
% Jan 22 2010
% Naval Research Laboratory
% 
% Description:  Convert Right Ascension/Declination angles in degrees to 
%               Altitude/Azimuth in degrees for an Alt/Az telescope mount.
%
% Input:    RA - Right Ascension in degrees
%           Dec - Declination in degrees
%           Latidute - Observer's Latitude (Negative for South) in degrees
%           Longitude - Observer's Longiture (Negative for West) in degrees
%           (optional) time - Date vector, as returned from 'clock.m', if 
%                             not supplied, the current date/time is used.
%           Jerome: time should be UT
% Output:   Altitude - Telescope Altitude in degrees
%           Azimuth - Telescope Azimuth in degrees
if nargin == 4
    time = clock;
    year = time(1);
    month = time(2);
    day = time(3);
    hour = time(4);
    min = time(5);
    sec = time(6);
else
    year = time(1);
    month = time(2);
    day = time(3);
    hour = time(4);
    min = time(5);
    sec = time(6);
end
JD = floor(365.25*(year + 4716.0)) + floor(30.6001*( month + 1.0)) + 2.0 - ...
    floor(year/100.0) + floor(floor(year/100.0 )/4.0) + day - 1524.5 + ...
    (hour + min/60 + sec/3600)/24;
D = JD - 2451543.5;
w = 282.9404 + 4.70935e-5*D;
M = mod(356.0470 + 0.9856002585*D,360);
L = w + M;
GMST0 = mod(L + 180,360)/15;
UT_hour = hour + min/60 + sec/3600;
SiderealTime = GMST0 + UT_hour + Longitude/15;
HourAngle = (SiderealTime*15 - RA);
A = cosd(HourAngle)*cosd(Dec)*cosd(90 - Latitude) - sind(Dec)*sind(90 - Latitude);
B = sind(HourAngle)*cosd(Dec);
C = cosd(HourAngle)*cosd(Dec)*sind(90 - Latitude) + sind(Dec)*cosd(90 - Latitude);
Az = atan2(B,A)*180/pi + 180;
Alt = asin(C)*180/pi;

