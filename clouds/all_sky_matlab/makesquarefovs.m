
function ptscarre=makesquarefovs(thtab,phitab)
ptscarre=[];


%transform alt-az center position to RADEC coords (needs to use date [2000 12 12 6 34 53] to align RA/az origins)
licklat=37.3;%deg

[RA Dec] = AltAz2RaDec(phitab/pi*180,thtab/pi*180,licklat,0,[2000 12 12 6 34 53])
phitab=Dec/180*pi;
thtab=RA/180*pi;

foc=0.6; %focal length (m)
detsz=1e-3*(25.8*2+1*(26.4-25.8))/16;% pixel size in m (3.2mm);
platesc=atan(detsz/foc) ;%angle (rad) per mm
platescpixdeg=(180/pi)*platesc %angle (deg) per mm
detarrside=32;
fovlensdeg=detarrside*platescpixdeg;



azimuth=thtab(1);
elevation=phitab(1);
r=atan(sqrt(2)*tan(fovlensdeg/2./180.*pi));
lat=elevation;
lat1=asin( sin(lat) * cos(r) + cos(lat)*sin(r)/sqrt(2) );
lat2=asin( sin(lat) * cos(r) - cos(lat)*sin(r)/sqrt(2) );

dlong1=acos( (cos(r) - sin(lat1) * sin(lat)) / ( cos(lat1) * cos(lat) ));
dlong2=acos( (cos(r) - sin(lat2) * sin(lat)) / ( cos(lat2) * cos(lat) ));

dlong1deg=dlong1/pi*180;
dlong2deg=dlong2/pi*180;

eldeg=180/pi*[lat2 ...
    lat2 ...
    lat1 ...
    lat1 ...
    ];
disp('eldeg:');disp(eldeg);

longdeg=[mod((azimuth-dlong2)*180./pi-180,360) ...
         mod((azimuth+dlong2)*180./pi-180,360) ...
         mod((azimuth+dlong1)*180./pi-180,360) ...
         mod((azimuth-dlong1)*180./pi-180,360) ...
        ];
    disp('longdeg:');disp(longdeg);
    if max(longdeg)-min(longdeg) >180
        longdeg(longdeg>180)=longdeg(longdeg>180)-360;
    end
     if min(longdeg) >180
        longdeg(longdeg>180)=longdeg(longdeg>180)-360;
    end
disp('longdeg:');disp(longdeg);


[Ax,Ay,Az]=sph2cart(longdeg(1)*pi/180,eldeg(1)*pi/180,1);
[Bx,By,Bz]=sph2cart(longdeg(2)*pi/180,eldeg(2)*pi/180,1);
[Cx,Cy,Cz]=sph2cart(longdeg(3)*pi/180,eldeg(3)*pi/180,1);
[Dx,Dy,Dz]=sph2cart(longdeg(4)*pi/180,eldeg(4)*pi/180,1);
A=[Ax Ay Az];
B=[Bx By Bz];
C=[Cx Cy Cz];
D=[Dx Dy Dz];


 ptscarre=[ptscarre;A;B;C;D]

 
 %%come back to altaz coords:
 Ry=roty((90-licklat)); Ry=Ry(1:3,1:3);
 ptscarre=ptscarre*Ry;
 
 
 
end


