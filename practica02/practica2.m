%% Limpiamos entorno

close all;
%clear all;
clear variables;
clear global;
clc;

%% Abrimos imágenes y las recortamos 

% Tamaño a utilizar en las imágenes
ren = 385;
col = 600;
tot = ren*col;

% Abrimos imágenes de entrenamiento como double
im1 = double(imread('im1.jpg'));
im2 = double(imread('im2.jpg'));
im3 = double(imread('im3.jpg'));

% Abrimos imágenes de entrenamiento como double
im4 = double(imread('im4.jpg'));

% Recortamos imagenes imágenes
im1 = im1(1:ren,1:col);
im2 = im2(1:ren,1:col);
im3 = im3(1:ren,1:col);
im4 = im4(1:ren,1:col);

%% Filtramos las imágenes con un filtro gaussiano

% imgaussfilt filtra la imagen con un kernel alisador 2-D de Gauss con
% desviación estándar de 0,5.
im1f = imgaussfilt(im1);
im2f = imgaussfilt(im2);
im3f = imgaussfilt(im3);

%% Creamos las mascaras sobre las imagenes filtradas

%%%% Mascaras de imágen 1
figure;imagesc(im1f);colormap(gray); axis square;
mask11=createMask(drawfreehand());
close all

figure;imagesc(im1f);colormap(gray); axis square;
mask21=createMask(drawfreehand());
close all

figure;imagesc(im1f);colormap(gray); axis square;
mask31=createMask(drawfreehand());
close all

%%%% Mascaras de imágen 2
figure;imagesc(im2f);colormap(gray); axis square;
mask12=createMask(drawfreehand());
close all

figure;imagesc(im2f);colormap(gray); axis square;
mask22=createMask(drawfreehand());
close all

figure;imagesc(im2f);colormap(gray); axis square;
mask32=createMask(drawfreehand());
close all

%%%% Mascaras de imágen 3
figure;imagesc(im3f);colormap(gray); axis square;
mask13=createMask(drawfreehand());
close all

figure;imagesc(im3f);colormap(gray); axis square;
mask23=createMask(drawfreehand());
close all

figure;imagesc(im3f);colormap(gray); axis square;
mask33=createMask(drawfreehand());
close all

%% Aplicamos las mascaras a las imágenes

% Obtenemos las regiones de la clase 1
r11 = immultiply(im1f,double(mask11));
r12 = immultiply(im2f,double(mask12));
r13 = immultiply(im3f,double(mask13));

% Obtenemos las regiones de la clase 2
r21 = immultiply(im1f,double(mask21));
r22 = immultiply(im2f,double(mask22));
r23 = immultiply(im3f,double(mask23));

% Obtenemos las regiones de la clase 3
r31 = immultiply(im1f,double(mask31));
r32 = immultiply(im2f,double(mask32));
r33 = immultiply(im3f,double(mask33));

%% Desplegamos imágenes filtradas

figure;

%%%% Mascaras de imágen 1
subplot(3, 3, 1);
imagesc(r11);colormap(gray); axis square; title('Corazón 1');

subplot(3, 3, 2);
imagesc(r21);colormap(gray); axis square; title('Halo 1');

subplot(3, 3, 3);
imagesc(r31);colormap(gray); axis square; title('Exterior 1');

%%%% Mascaras de imágen 1
subplot(3, 3, 4);
imagesc(r12);colormap(gray); axis square; title('Corazón 2');

subplot(3, 3, 5);
imagesc(r22);colormap(gray); axis square; title('Halo 2');

subplot(3, 3, 6);
imagesc(r32);colormap(gray); axis square; title('Exterior 2');

%%%% Mascaras de imágen 1
subplot(3, 3, 7);
imagesc(r13);colormap(gray); axis square; title('Corazón 3');

subplot(3, 3, 8);
imagesc(r23);colormap(gray); axis square; title('Halo 3');

subplot(3, 3, 9);
imagesc(r33);colormap(gray); axis square; title('Exterior 3');

%% Calculamos la probabilidad de cada clase

% Calculamos la probabilidad de cada región/clase
probC1 = (sum(mask11(:))+sum(mask12(:))+sum(mask13(:)))/(3*tot);
probC2 = (sum(mask21(:))+sum(mask22(:))+sum(mask23(:)))/(3*tot);
probC3 = (sum(mask31(:))+sum(mask32(:))+sum(mask33(:)))/(3*tot);

% Solo por verificar, la probabilidad tendria que ser 1
probT = probC1 + probC2 + probC3;

%% Concatenamos las regiones en una sola matriz para cada clase

% Matriz de las regiones de la clase 1
mr1 = cat(1,r11,r12,r13);

% Matriz de las regiones de la clase 2
mr2 = cat(1,r21,r22,r23);

% Matriz de las regiones de la clase 3
mr3 = cat(1,r31,r32,r33);

%% Calculamos la media de cada clase

% Creamos las matrices para las medias
med1 = zeros([3 1]);
med2 = zeros([3 1]);
med3 = zeros([3 1]);

% Calculamos la media para la clase 1
cont = 0;
for s=1:3
    for x=1:ren
        for y=1:col
            if( mr1(((ren * (s-1)) + x),y) ~= 0)
                med1(1) = med1(1) + mr1(((ren * (s-1)) + x),y);
                med1(2) = med1(2) + x;
                med1(3) = med1(3) + y;
                cont = cont + 1;
            end
        end
    end
end
med1 = med1 / cont;

% Calculamos la media para la clase 2
cont = 0;
for s=1:3
    for x=1:ren
        for y=1:col
            if( mr2(((ren * (s-1)) + x),y) ~= 0)
                med2(1) = med2(1) + mr2(((ren * (s-1)) + x),y);
                med2(2) = med2(2) + x;
                med2(3) = med2(3) + y;
                cont = cont + 1;
            end
        end
    end
end
med2 = med2 / cont;

% Calculamos la media para la clase 3
cont = 0;
for s=1:3
    for x=1:ren
        for y=1:col
            if( mr3(((ren * (s-1)) + x),y) ~= 0)
                med3(1) = med3(1) + mr3(((ren * (s-1)) + x),y);
                med3(2) = med3(2) + x;
                med3(3) = med3(3) + y;
                cont = cont + 1;
            end
        end
    end
end
med3 = med3 / cont;

%% Calculamos la covarianza de cada clase

% Creamos las matrices para las covarianzas
cov1 = zeros(3);
cov2 = zeros(3);
cov3 = zeros(3);

% Calculamos la matriz de covarianza para la clase 1
for s=1:3
    for x=1:ren
        for y=1:col
            if( mr1(((ren * (s-1)) + x),y) ~= 0)
                % Diagonal para la clase 1
                cov1(1, 1) = cov1(1, 1) + ((mr1(((ren * (s-1)) + x),y) - med1(1)) ^2);
                cov1(2, 2) = cov1(2, 2) + (((ren * (s-1)) - med1(2)) ^2);
                cov1(3, 3) = cov1(3, 3) + ((y - med1(3)) ^2);
                
                %  Triangulo superior para la clase 1
                cov1(1, 2) = cov1(1, 2) + ( (mr1(((ren * (s-1)) + x),y) - med1(1)) * ((ren * (s-1)) - med1(2)) );
                cov1(1, 3) = cov1(1, 3) + ( (mr1(((ren * (s-1)) + x),y) - med1(1)) * (y - med1(3)) );
                cov1(2, 3) = cov1(2, 3) + ( ((ren * (s-1)) - med1(2)) * (y - med1(3)) );
            end
        end
    end
end

% Triangulo inferior para la clase 1
cov1(2, 1) = cov1(1, 2);
cov1(3, 1) = cov1(1, 3);
cov1(3, 2) = cov1(2, 3);

% Calculamos la matriz de covarianza para la clase 2
for s=1:3
    for x=1:ren
        for y=1:col
            if( mr2(((ren * (s-1)) + x),y) ~= 0)
                % Diagonal para la clase 2
                cov2(1, 1) = cov2(1, 1) + ((mr2(((ren * (s-1)) + x),y) - med2(1)) ^2);
                cov2(2, 2) = cov2(2, 2) + (((ren * (s-1)) - med2(2)) ^2);
                cov2(3, 3) = cov2(3, 3) + ((y - med2(3)) ^2);
                
                %  Triangulo superior para la clase 2
                cov2(1, 2) = cov2(1, 2) + ( (mr2(((ren * (s-1)) + x),y) - med2(1)) * ((ren * (s-1)) - med2(2)) );
                cov2(1, 3) = cov2(1, 3) + ( (mr2(((ren * (s-1)) + x),y) - med2(1)) * (y - med2(3)) );
                cov2(2, 3) = cov2(2, 3) + ( ((ren * (s-1)) - med2(2)) * (y - med2(3)) );
            end
        end
    end
end

% Triangulo inferior para la clase 2
cov2(2, 1) = cov2(1, 2);
cov2(3, 1) = cov2(1, 3);
cov2(3, 2) = cov2(2, 3);

% Calculamos la matriz de covarianza para la clase 3
for s=1:3
    for x=1:ren
        for y=1:col
            if( mr3(((ren * (s-1)) + x),y) ~= 0)
                % Diagonal para la clase 3
                cov3(1, 1) = cov3(1, 1) + ((mr3(((ren * (s-1)) + x),y) - med3(1)) ^2);
                cov3(2, 2) = cov3(2, 2) + (((ren * (s-1)) - med3(2)) ^2);
                cov3(3, 3) = cov3(3, 3) + ((y - med3(3)) ^2);
                
                %  Triangulo superior para la clase 3
                cov3(1, 2) = cov3(1, 2) + ( (mr3(((ren * (s-1)) + x),y) - med3(1)) * ((ren * (s-1)) - med3(2)) );
                cov3(1, 3) = cov3(1, 3) + ( (mr3(((ren * (s-1)) + x),y) - med3(1)) * (y - med3(3)) );
                cov3(2, 3) = cov3(2, 3) + ( ((ren * (s-1)) - med3(2)) * (y - med3(3)) );
            end
        end
    end
end

% Triangulo inferior para la clase 1
cov3(2, 1) = cov3(1, 2);
cov3(3, 1) = cov3(1, 3);
cov3(3, 2) = cov3(2, 3);

%% Calculamos el termino logaritmico para cada clase