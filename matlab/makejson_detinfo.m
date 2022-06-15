close all 
clear all

load('/Users/jeromemaire/Documents/SETI/PANOSETI/code_reduce/json/HamamatsuVop.mat');
  
JSONFILE_name= 'detector_info.json'; 
    fid=fopen(JSONFILE_name,'w') 
 fprintf(fid, '%s\n','['); 
allserial=isfinite(detV25C(:,1));
allserialind=find(allserial);
for ii=1:numel(allserialind)
 inddet1=allserialind(ii);
 HV1 = mean(detV25C(inddet1:inddet1+63,2));

 
    
    s = struct("serialno", detV25C(inddet1,1), "operating_voltage", num2str(HV1,'%3.3f')); 
    encodedJSON = jsonencode(s); 
   if ii~=numel(allserialind)
       fprintf(fid, '%s,\n',encodedJSON); 
   else
         fprintf(fid, '%s\n',encodedJSON);
   end
end
 fprintf(fid, '%s\n',']'); 
fclose('all'); 
disp('Done')