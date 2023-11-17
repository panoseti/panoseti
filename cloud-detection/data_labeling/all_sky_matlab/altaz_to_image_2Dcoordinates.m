% This function finds the coordinates of PANOSETI field-of-view corners in
% Lick webcam images.
% It is assuming Lick webcam AllSky images are 480x640 pixels
% and skycam2 images are 521x765 pixels
%
% Inputs: 
%  webcamname: can be 'Sky' or 'Sky2'
%  dateproc: date of  observations, format: yyyyMMdd example: dateproc='20221106'
%
% Outputs:
%  xpanocorner1, ypanocorner1: x,y coordinates of left-bottom corner of
%  panoseti field-of-view in Lick webcam all-sky image
%  xpanocorner2, ypanocorner2: x,y coordinates of right-top corner of
%  panoseti field-of-view in Lick webcam all-sky image
% These coords are given with the origin at the top-left corner of the Lick
% webcam image, with y-axis going down and x-axis going right
%
% JM 2023-05 started.
%
function [xcorners,ycorners] = altaz_to_image_2Dcoordinates(webcamname, dateproc)



% load astrometric database
load('astrom.mat')
%find the prior closest astrometry in time
% Convert date strings to datetime objects
dates = datetime((table2array(astrom(:, 1))), 'InputFormat', 'yyyy/MM/dd HH:mm:ss');

% reformat User-defined date
user_date = datetime(dateproc, 'InputFormat', 'yyyyMMdd');

% Find the row with the prior closest date
[~, idx] = min(abs(dates - user_date));

% Display the row with the prior closest date
prior_closest_date_row = dates(idx);
disp(['Prior closest astrom date: ', char(prior_closest_date_row)]);
astromAlt=table2array(astrom(idx,3))
astromAz=table2array(astrom(idx,4))

%  astromAlt=70
%  astromAz=10

ptscarre=makesquarefovs(astromAz/180.*pi, astromAlt/180.*pi);


%fix skycam misorientation of N/S and up/down axis (~18deg)
if ~strcmp(webcamname,'SC2') %for SC:
 Rz=rotz(-18); Rz=Rz(1:3,1:3);
 ptscarre=ptscarre*Rz;
end


if ~strcmp(webcamname,'SC2') %for SC:
    yzenith=385 %zenith coordinates in pixels
    xzenith=263 %zenith coordinates in pixels
    pix2pi=630/2/90;%angular scale in pixel per degree
else %for SC2:
    yzenith=325 %zenith coordinates in pixels
    xzenith=247 %zenith coordinates in pixels
    pix2pi=480/2/90;
end

xcorners=xzenith+90.*pix2pi*ptscarre(:,1) %factor 90 to take into account the full extent of altitude range in the image
ycorners=yzenith-90.*pix2pi*ptscarre(:,2) %factor 90 for full extent of altitude range in the image


% disp('ptscarre:')
% disp(ptscarre)
end