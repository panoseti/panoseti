% This routine overplots pano field-of-view (fov) on top of Lick Allsky and
% skycam2 images. Equatorial mounts are assumed for Panoseti telescopes regarding the field-of-view
% orientation (Position Angle), also working for alt-az mounts observing at meridian 
% 
%
%close all
%clear all
%% Modify the following parameters:
webcrep='SC_imgs/'; % the directory of the Lick webcam images
choosewebcam=1; %=1 reduces Lick AllSky images (skycam2); =2 reduces Lick SkyCam images
dateproc='20230801'; % to search the proper panoseti astrometry, enter the date of the Lick webcam images as yyyyMMdd of the 
%%%

if choosewebcam==1
webcamname='SC'; %skycam
    fwebc=dir([webcrep  '*' webcamname '_*']);
elseif choosewebcam==2 %skycam2
    webcamname='SC2';%skycam2 Lick
    fwebc=dir([webcrep  '*' webcamname '*']);
end


% loop over the webcam images and make a figure of the image with Panoseti
% field-of-view
firstimc=1;
tic
for ii=firstimc:5  %size(fwebc,1)
    
    figure('Position',[40 40 1200 800],'Color','w')
   
    imawc=imread([webcrep fwebc(ii).name]);
    
    hold on

    him=imagesc(imawc);
    
    set(gca,'YDir','Reverse')
    axis image
    
    
    %%%% Manually hard-coded coordinates corresponding to Pano fov in 2023-09:
    if choosewebcam==2
 %Allsky Lick skycam2
 %location of zenith in pixel (assuming origin in top-left corner of the lick webcam image, x-axis going down, y-axis going right)
        cx1=278;
        cx2=318; %1..480
        cy1=310;
        cy2=352;%1..640
        decy=0;
elseif choosewebcam==1
%skycam 
        cx1=293;
        cx2=343; %1..521
        cy1=340;
        cy2=405;%1..765
        decy=14;
end
    plot([cy1+decy cy1],[cx1 cx2],'r-','Linewidth',3)
    plot([cy2 cy2-decy],[cx1 cx2]+decy,'r-','Linewidth',3)
    plot([cy1+decy cy2],[cx1 cx1+decy],'r-','Linewidth',3)
    plot([cy1 cy2-decy],[cx2 cx2+decy],'r-','Linewidth',3)
    
    
     %%%% automated calculations of Pano fov location:
     [xcorners,ycorners] = altaz_to_image_2Dcoordinates(webcamname, dateproc)
  
    plot([ycorners(1) ycorners(2)],[xcorners(1) xcorners(2)],'g--','Linewidth',3)
    plot([ycorners(2) ycorners(3)],[xcorners(2) xcorners(3)],'g-.','Linewidth',3)
    plot([ycorners(3) ycorners(4)],[xcorners(3) xcorners(4)],'g-','Linewidth',3)
    plot([ycorners(4) ycorners(1)],[xcorners(4) xcorners(1)],'g-','Linewidth',3)
    
    ti=title(['Lick ' webcamname '  ' char(fwebc(ii).name)] );

    set(ti,'Interpreter','none')
    
    
    drawnow
    
end