function [RA Dec] = AltAz2RaDec(Altitude,Azimuth,Latitude,Longitude,time)
% By: Christopher Wilcox and Ty Martinez
% Jan 22 2010
% Naval Research Laboratory
% 
% Description:  Convert Altitude/Azimuth angles in degrees to Right
%               Ascension/Declination in degrees for an Alt/Az telescope 
%               mount.
%
% Input:    Altitude - Telescope Altitude in degrees
%           Azimuth - Telescope Azimuth in degrees
%           Latidute - Observer's Latitude (Negative for South) in degrees
%           Longitude - Observer's Longiture (Negative for West) in degrees
%           (optional) time - Date vector, as returned from 'clock.m', if 
%                             not supplied, the current date/time is used.
% Output:   RA - Right Ascension in degrees
%           Dec - Declination in degrees 
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
T = (JD - 2451545)/36525;
ThetaGMST = 67310.54841 + (876600*3600 + 8640184.812866)*T + .093104*(T^2) - (6.2*10^-6)*(T^3);
ThetaGMST = mod((mod(ThetaGMST,86400*(ThetaGMST/abs(ThetaGMST)))/240),360);
ThetaLST = ThetaGMST + Longitude;
ThetaLST = mod(ThetaLST,360);
Dec = asind(sind(Altitude)*sind(Latitude) + cosd(Altitude)*cosd(Latitude)*cosd(Azimuth));
cos_RA = (sind(Altitude) - sind(Dec)*sind(Latitude))/(cosd(Dec)*cosd(Latitude));
RA = acosd(cos_RA);
if sind(Azimuth) > 0 
    RA = 360 - RA;
end
RA = ThetaLST - RA;
if RA < 0
    RA = RA + 360;
end
if Dec >= 0
    Dec = abs(Dec);
else
    Dec = -abs(Dec);
end