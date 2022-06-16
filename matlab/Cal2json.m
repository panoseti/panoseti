clear;
clc;
close all;

% load the calibration tables
CalibrationDB = load(['.' filesep 'CalibrationDB.mat']);
gainmap_inc= load(['.' filesep 'gainmap_inc.mat']);

% get the size of quaboDETtable
% in this table the meanings of columns are:
% quabo_SN detector0_SN detector1_SN detector2_SN detector3_SN unknown a0 a1 a2 a3 b0 b1 b2 b3
[r,c] = size(CalibrationDB.quaboDETtable);

for i = 1:r
    % create json file name
    fname = ['quabo_calib_',int2str(CalibrationDB.quaboDETtable(i,1)),'.json'];
    % get quabo_SN from calibration file
    content.pixel_gain = gainmap_inc.gainmapallgmeanSN(:,:,i);
    % get detector_SN, a and b from calibration file
    for j = 1:4
        det(j).detserial = CalibrationDB.quaboDETtable(i, j+1); 
        det(j).a = CalibrationDB.quaboDETtable(i, j+6);
        det(j).b = CalibrationDB.quaboDETtable(i, j+10);
        det(j).m = 0.08; % m and n are fixed value
        det(j).n = 12.0; % m and n are fixed value
    end
    content.quadrants = [det(1), det(2), det(3), det(4)];
    json = jsonencode(content,"PrettyPrint",true)
    % write json content to a json file
    fid = fopen(fname, 'wt');
    fprintf(fid, '%s',json);
    fclose(fid);
end
