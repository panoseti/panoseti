close all 
clear all


 load((['/Users/jeromemaire/Documents/SETI/PANOSETI/code_reduce/json/' 'CalibrationDB.mat']))
  [ig,indexquabosn]=ismember(['QuaboSN'] ,quaboconfig);
 inddetrow=find(quaboDETtable(:,1)==str2num(cell2mat(quaboconfig(indexquabosn,3))));
A0=gain*quaboDETtable(inddetrow,7);
A1=gain*quaboDETtable(inddetrow,8);
A2=gain*quaboDETtable(inddetrow,9);
A3=gain*quaboDETtable(inddetrow,10);
B0=quaboDETtable(inddetrow,11);
B1=quaboDETtable(inddetrow,12);
B2=quaboDETtable(inddetrow,13);
B3=quaboDETtable(inddetrow,14);



load('/Users/jeromemaire/Documents/SETI/PANOSETI/code_reduce/json/HamamatsuVop.mat');
  
JSONFILE_name= 'detector_info.json'; 
    fid=fopen(JSONFILE_name,'w') 

allserial=isfinite(detV25C(:,1));
allserialind=find(allserial);
for ii=1:numel(allserialind)
 inddet1=allserialind(ii);
 HV1 = mean(detV25C(inddet1:inddet1+63,2));

 
    
    s = struct("serialno", detV25C(inddet1,1), "operating_voltage", num2str(HV1,'%3.3f')); 
    encodedJSON = jsonencode(s); 
    fprintf(fid, '%s,\n',encodedJSON); 
end
fclose('all'); 
disp('Done')