close all 
clear all

load((['/Users/jeromemaire/Documents/SETI/PANOSETI/code_reduce/json/' 'Panoconfig2.mat']))
  
JSONFILE_name= 'quabo_info.json'; 
 fid=fopen(JSONFILE_name,'w') 
 fprintf(fid, '%s\n','['); 

 
% %allserial=isfinite(detV25C(:,1));
% allserialind=find(allserial);
for ii=1:4 %over modules
    for jj=1:4 % over quads
%  inddet1=allserialind(ii);
%  HV1 = mean(detV25C(inddet1:inddet1+63,2));
qq=(jj-1)*6;
    
 ssub={string(cell2mat(config2(ii,5+qq))) ; string(cell2mat(config2(ii,6+qq))) ; string(cell2mat(config2(ii,7+qq))) ; string(cell2mat(config2(ii,8+qq))) }
 encodedJSONsub = jsonencode(ssub)
 %cell2mat(config2(ii,4))
    s = struct("uid", ' ', ...
        "serialno", cell2mat(config2(ii,4+qq)),...
        "board_version", ' ', ...
        "detector_serialno",  encodedJSONsub); 
    encodedJSON = jsonencode(s); 
   if ii==4 && jj==4 %numel(allserialind)
        fprintf(fid, '%s\n',encodedJSON);
     
   else
          fprintf(fid, '%s,\n',encodedJSON); 
   end
    end
end
 fprintf(fid, '%s\n',']'); 
fclose('all'); 
disp('Done')