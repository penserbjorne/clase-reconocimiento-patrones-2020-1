close all
clear all
clc

im = imread('imagen.png');
im_g = rgb2gray(im);

filtro5=fspecial('gaussian',[5,5]);
filtro9=fspecial('gaussian',[9,9]);
filtro11=fspecial('gaussian',[11,11]);
imFilG5=imfilter(im_g,filtro5);
imFilG9=imfilter(im_g,filtro9);
imFilG11=imfilter(im_g,filtro11);
imFilM=medfilt2(im_g);

mediaIntensidad=mean(imFilM(:));
im_bin= imFilM > mediaIntensidad;

[ren,col]=size(im);
im_r=imcrop(im_bin,[288,300,90,70]);

figure,title('Filtros')
subplot(2,4,1),imshow(im),title('Original')
subplot(2,4,2),imshow(im_g),title('Escala de grises')
subplot(2,4,3),imshow(imFilG5),title('Gausiano 5X5')
subplot(2,4,4),imshow(imFilG9),title('Gausiano 9X9')
subplot(2,4,5),imshow(imFilG11),title('Gausiano 11X11')
subplot(2,4,6),imshow(imFilM),title('Media no lineal')
subplot(2,4,7),imshow(im_bin),title('Binaria')
subplot(2,4,8),imshow(im_r),title('Binaria recortada')

