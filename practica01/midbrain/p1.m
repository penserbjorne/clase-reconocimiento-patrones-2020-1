close all
clear all
clc

im = imread('imagen.png');
im_g = rgb2gray(im);

filtro=fspecial('gaussian',[5,5]);
imFilG=imfilter(im_g,filtro);
imFilM=medfilt2(im_g);

mediaIntensidad=mean(imFilM(:));
im_bin= imFilM > mediaIntensidad;

figure,
subplot(1,4,1),imshow(im_g)
subplot(1,4,2),imshow(imFilG)
subplot(1,4,3),imshow(imFilM)
subplot(1,4,4),imshow(im_bin)

