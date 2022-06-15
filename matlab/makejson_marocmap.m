close all 
clear all



        
 load((['/Users/jeromemaire/Documents/SETI/PANOSETI/code_reduce/json/' 'MarocMap.mat']))
 

%load('/Users/jeromemaire/Documents/SETI/PANOSETI/code_reduce/json/HamamatsuVop.mat');
  



    JSONFILE_name= ['quabo_pixelmap_maroc2phys_qfp.json']; 
    fid=fopen(JSONFILE_name,'w') 

    s = struct("pixel_map_maroc2phys", marocmap-1); 
    encodedJSON = jsonencode(s); 
    fprintf(fid, '%s,\n',encodedJSON); 
     fclose('all'); 
     
         JSONFILE_name= ['quabo_pixelmap_phys2maroc_qfp.json']; 
    fid=fopen(JSONFILE_name,'w') 

    s = struct("pixel_map_phys2maroc", marocmap16-1); 
    encodedJSON = jsonencode(s); 
    fprintf(fid, '%s,\n',encodedJSON); 
     fclose('all'); 
     
     %%%%BGA:
     
        
 load((['/Users/jeromemaire/Documents/SETI/PANOSETI/code_reduce/json/' 'MarocMapBGA.mat']))
 
    JSONFILE_name= ['quabo_pixelmap_maroc2phys_bga.json']; 
    fid=fopen(JSONFILE_name,'w') 

    s = struct("pixel_map_maroc2phys", marocmap-1); 
    encodedJSON = jsonencode(s); 
    fprintf(fid, '%s,\n',encodedJSON); 
     fclose('all'); 
     
         JSONFILE_name= ['quabo_pixelmap_phys2maroc_bga.json']; 
    fid=fopen(JSONFILE_name,'w') 

    s = struct("pixel_map_phys2maroc", marocmap16-1); 
    encodedJSON = jsonencode(s); 
    fprintf(fid, '%s,\n',encodedJSON); 
     fclose('all'); 
     
     
disp('Done')