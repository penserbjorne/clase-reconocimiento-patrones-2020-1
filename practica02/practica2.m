close all
clear all
clc

%Variables a utilizar
ren = 385;
col = 600;
tot = ren*col;

%Abrir imágenes entrenamiento como double
im1 = double(imread('im1.jpg'));
im2 = double(imread('im2.jpg'));
im3 = double(imread('im3.jpg'));

%Abrir imagen a clasificar como double
im4 = double(imread('im4.jpg'));

%Se recortan las imágenes
im1 = im1(1:ren,5:col);
im2 = im2(1:ren,5:col);
im3 = im3(1:ren,5:col);
im4 = im4(1:ren,5:col);

%Se filtran las imágenes
g = fspecial('gaussian',100,5);
im1f = imfilter(im1,g,'symmetric','conv');
im2f = imfilter(im2,g,'symmetric','conv');
im3f = imfilter(im3,g,'symmetric','conv');

%Se abren imágenes para separar clases
figure;imagesc(im1);colormap(gray)
mask11=createMask(drawfreehand());
close all

figure
imagesc(im1);colormap(gray)
mask21=createMask(drawfreehand());
close all

figure;imagesc(im1);colormap(gray)
mask31=createMask(drawfreehand());
close all

figure;imagesc(im2);colormap(gray)
mask12=createMask(drawfreehand());
close all

figure
imagesc(im2);colormap(gray)
mask22=createMask(drawfreehand());
close all

figure;imagesc(im2);colormap(gray)
mask32=createMask(drawfreehand());
close all

figure;imagesc(im3);colormap(gray)
mask13=createMask(drawfreehand());
close all

figure
imagesc(im3);colormap(gray)
mask23=createMask(drawfreehand());
close all

figure;imagesc(im3);colormap(gray)
mask33=createMask(drawfreehand());
close all

%Se obtienen las regiones clase 1

r11 = immultiply(im1f,double(mask11));
r12 = immultiply(im2f,double(mask12));
r13 = immultiply(im3f,double(mask13));

%Matriz de todas las regiones clase 1
mr1 = cat(1,r11,r12,r13);

%Se obtienen las regiones clase 2

r21 = immultiply(im1f,double(mask21));
r22 = immultiply(im2f,double(mask22));
r23 = immultiply(im3f,double(mask23));

%Matriz de todas las regiones clase 2
mr2 = cat(1,r21,r22,r23);

%Se obtienen las regiones clase 3

r31 = immultiply(im1f,double(mask31));
r32 = immultiply(im2f,double(mask32));
r33 = immultiply(im3f,double(mask33));

%Matriz de todas las regiones clase 3
mr3 = cat(1,r31,r32,r33);

%Probabilidad de cada región
probC1 = (sum(mask11(:))+sum(mask12(:))+sum(mask13(:)))/(3*tot);
probC2 = (sum(mask21(:))+sum(mask22(:))+sum(mask23(:)))/(3*tot);
probC3 = (sum(mask31(:))+sum(mask32(:))+sum(mask33(:)))/(3*tot);

probT = probC1 + probC2 + probC3;

% Calculamos la media y la covarianza
meanR = zeros(3,1);
covaR = zeros(3,3);

for s=1:3
    for x=1:(ren-1)
        for y=1:(col-5)
            if(mr1(ren*(s-1)+x,y) ~= 0)
                meanR(1) = meanR(1) + mr1(ren*(s-1)+x,y);
                meanR(2) = meanR(2) + x;
                meanR(3) = meanR(3) + y;
            end
        end
    end
end


for s=1:3
    for x=1:(ren-1)
        for y=1:(col-5)
            if(mr2(ren*(s-1)+x,y) ~= 0)
                meanR(1) = meanR(1) + mr2(ren*(s-1)+x,y);
                meanR(2) = meanR(2) + x;
                meanR(3) = meanR(3) + y;
            end
        end
    end
end


for s=1:3
    for x=1:(ren-1)
        for y=1:(col-5)
            if(mr3(ren*(s-1)+x,y) ~= 0)
                meanR(1) = meanR(1) + mr3(ren*(s-1)+x,y);
                meanR(2) = meanR(2) + x;
                meanR(3) = meanR(3) + y;
            end
        end
    end
end

meanR = meanR / (3*tot);















