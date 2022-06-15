close all 
clear all


 load((['/Users/jeromemaire/Documents/SETI/PANOSETI/code_reduce/json/' 'CalibrationDB.mat']))
 
 load(['/Users/jeromemaire/Documents/SETI/PANOSETI/code_reduce/json/' 'gainmap_mean_.mat'])

        load(['/Users/jeromemaire/Documents/SETI/PANOSETI/code_reduce/json/' 'gainmap_inc.mat'])
    

%load('/Users/jeromemaire/Documents/SETI/PANOSETI/code_reduce/json/HamamatsuVop.mat');
  


for inddetrow=15:15 %1:size(quaboDETtable,2)
    % [ig,indexquabosn]=ismember(['QuaboSN'] ,quaboconfig);
%find(quaboDETtable(:,1)==str2num(cell2mat(quaboconfig(indexquabosn,3))));
QuaboSN=quaboDETtable(inddetrow,1);


    JSONFILE_name= ['quabo_calib_' num2str((QuaboSN)) '.json']; 
    fid=fopen(JSONFILE_name,'w') 
q1=quaboDETtable(inddetrow,2);
q2=quaboDETtable(inddetrow,3);
q3=quaboDETtable(inddetrow,4);
q4=quaboDETtable(inddetrow,5);
A0g=quaboDETtable(inddetrow,7);
A1g=quaboDETtable(inddetrow,8);
A2g=quaboDETtable(inddetrow,9);
A3g=quaboDETtable(inddetrow,10);
B0=quaboDETtable(inddetrow,11);
B1=quaboDETtable(inddetrow,12);
B2=quaboDETtable(inddetrow,13);
B3=quaboDETtable(inddetrow,14);
Mp=0.08;
N=12.0;
    coeffs1=struct("detserial",q1,"a", A0g,"b",B0,"m", Mp, "n",N);
    coeffs2=struct("detserial",q2,"a", A1g,"b",B1,"m", Mp, "n",N);
    coeffs3=struct("detserial",q3,"a", A2g,"b",B2,"m", Mp, "n",N);
    coeffs4=struct("detserial",q4,"a", A3g,"b",B3,"m", Mp, "n",N);
    s = struct("quadrants", coeffs1); 
    encodedJSON = jsonencode(s); 
    fprintf(fid, '%s,\n',encodedJSON); 
      s = struct("quadrants", coeffs2); 
    encodedJSON = jsonencode(s); 
    fprintf(fid, '%s,\n',encodedJSON); 
      s = struct("quadrants", coeffs3); 
    encodedJSON = jsonencode(s); 
    fprintf(fid, '%s,\n',encodedJSON); 
      s = struct("quadrants", coeffs4); 
    encodedJSON = jsonencode(s); 
    fprintf(fid, '%s,\n',encodedJSON); 
    
    
    %%%%%%
    quaboconfig_gain=gainmapallgmeanSN(:,:,inddetrow);

    s = struct("pixel_gain", quaboconfig_gain); 
    encodedJSON = jsonencode(s); 
    fprintf(fid, '%s,\n',encodedJSON); 
end
fclose('all'); 
disp('Done')