close all
clear all
clc 
%Se carga la imagen
im = imread('imagen.png');
%Se convierte la imagen a escala de grises
im_g = rgb2gray(im);
%Se filtra la imagen con un filtro de media en 2d
im_f = medfilt2(im_g);
%Se ajusta el contraste de la imagen
im_a = imadjust(im_f);
%Se obtiene el tamaño de la matriz de la imagen
[ncols,nrows]=size(im_a);
%Se despliega la imagen para obtener la posici'on de 
%los pixeles semilla 
imshow(im_a)
%Se guardan los valores de los pixeles
[x,y] = getpts;
%Se convierte a valores enteros para usar como par'ametros
xi=int32(x);
yi=int32(y);
%Se cierra la figura creada
close all
%Se binariza la imagen. Calcule el umbral de imagen adaptable
%localmente elegido utilizando estadísticas de imagen 
%de primer orden locales alrededor de cada píxel. 
BW = imbinarize(im_a, 'adaptive');
%Se convierten los valores obtenidos a enteros.
bin = im2uint8(BW);
%Se crea una m'ascara binaria para los pixeles
%con intensidad de gris similar con una tolerancia de 1
bw = grayconnected(bin,xi,yi,1);
%Se obtiene el area total
area = bwarea(bw);
%Se recorta la imagen
im_r = imcrop(bw,[285,300,95,70]);
%Se muestran los resultados
figure
subplot(3,2,1),imshow(im_g),title('Original')
subplot(3,2,2),imhist(im_g,128),title('Histograma de original')
subplot(3,2,3),imshow(im_a),title('Filtrada y ajustada')
subplot(3,2,4),imhist(im_a,128),title('Histograma filtrado y ajustado')
subplot(3,2,5),imshow(bw),title('Mesencéfalo')
subplot(3,2,6),imshow(im_r),title('Imagen recortada')