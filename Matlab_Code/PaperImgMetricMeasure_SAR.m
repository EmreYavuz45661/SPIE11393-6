dr = 'C:\Users\eyavu\OneDrive\Documents\Programming\Project Workspace\Code and Txt Files\Python_Sean\codes\Data_Generative_Code\PreTrained_Networks\Paper_Img';
addpath(dr);
close all
clear all

%Load image and convert to double format
im_saved = imread('SAR_Recon_UNET.png');
im_g = double(rgb2gray(im_saved));
img_unet = im_g ./ (max(max(im_g)));

im_saved = imread('ground_truth_495.jpg');
im_g = double(im_saved);
img_true = im_g ./ (max(max(im_g)));;

im_2 = imread('noisy_images_495.jpg');
im_2_double = double(im_2);
img_input = im_2_double ./ (max(max(im_2_double)));;



figure(1);
subplot(3,1,1);imagesc(img_input);title({"Input Image to U-Net ";"(Image No: 495)"});grid on;colorbar;
subplot(3,1,2);imagesc(img_unet);title({"U-Net Output";"(Image No: 495)"});grid on;colorbar;
subplot(3,1,3);imagesc(img_true);title({"Ground Truth";"(Image No: 495)"});grid on;colorbar;
saveas(gcf,[dr,'\ResultsSampleImg495.png'])

figure(2);
subplot(3,1,1);imagesc(img_input(90:120, 45:75));title({"Input Image to U-Net";"(Zoomed in to the Region of Interest)"});grid on;colorbar;
subplot(3,1,2);imagesc(img_unet(90:120, 45:75));title({"U-Net Output (Zoomed Region)";"(Zoomed in to the Region of Interest)"});grid on;colorbar;
subplot(3,1,3);imagesc(img_true(90:120, 45:75));title({"Ground Truth (Zoomed Region)";"(Zoomed in to the Region of Interest)"});grid on;colorbar;
saveas(gcf,[dr,'\ResultsSampleZoomedImg495.png'])




%RNN use of iradon
%im_g = iradon(im_g, 0:1:179,128);

%Read noisy image, convert to double, and then scale by max
im_2 = imread('noisy_images_495.jpg');
im_2_double = double(im_2);
im_2_double = im_2_double/max(max(im_2_double));
im_2 = im_2/max(max(im_2));

%Scale Recon or Truth image by max
im = im_g/max(max(im_g));
%EDGE SPREAD
x2 = 79;
y2 = 110;
Edge_Block = im(x2:x2+6,y2:y2+6);
Edge_Block = Edge_Block/max(max(Edge_Block));
%View Region (Remove for True results)
%im(x2:x2+6,y2:y2+6) = 10;
[Gmag,Gdir] = imgradient(Edge_Block);
Gradient_Average = mean(Gmag(:));
Std = std(Gmag(:),1);
Edge_Spread = Gradient_Average/Std

%SNR
RMS = @(x) sqrt(mean(x.^2));
SNR = (RMS(im) / RMS(im_2_double)) ^ 2

%CNR
x_ref = 32:42;
y_ref = 70:80;
x_box = (83:93);
y_box = (108:118);

ref = im(x_ref,y_ref);
ref_m = mean(mean(ref));
ref_var = var(var(ref));

box = im(x_box,y_box);
box_m = mean(mean(box));
box_var = var(var(box));

CNR = abs(box_m-ref_m)/sqrt(box_var+ref_var)

%Show image
figure(1);
imagesc(im);colorbar;title('U-Net');

%Measure image edges along a graph
%figure(2);
%plot(im(:,110));title('RNN');hold on;
%plot(im(:,111));title('RNN');
%plot(im(:,112));title('RNN');




%Not useful, used for past RGB formats
%figure(3);
%imagesc(im);colorbar;
%figure(4);
%imagesc(im(:,:,1));colorbar;
%figure(5);
%imagesc(im(:,:,2));colorbar;
%figure(6);
%imagesc(im(:,:,3));colorbar;

%Shows edge spread visually
%figure;
%imagesc(Gmag/max(max(Gmag))); colorbar;

    

