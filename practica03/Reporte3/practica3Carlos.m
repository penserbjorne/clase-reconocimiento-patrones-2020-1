%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% LDA %%%%
%%%%%%%%%%%%%%%%%%%
%% Both Machine learning and bionfirmatics are required to run the code
%% Written and mantained by Simon, Carlos and Paul

%% Cleaning the environment
clear all;
close all;
clc
%% Define the number of samples 
samples = 5;
level = 10;
line = 1;
%% Perfoming analysis using degrees and distance
distV = 8;
distH = 8;
%% Image feature extraction (using D6 image)
for x=1:samples
    for y=1:samples
        imageX = imread('D6.BMP'); %
        imageX = rgb2gray(imageX);  % RGB TO GRAY
        imageX = imageX(1*x:64*x,1*y:64*y); % 64*64
        imageX = histeq(imageX, level); %enhance contrast using histogram equalization
        cocurrenceM = zeros(level);
        %Feature extraction
        cocurrenceM = graycomatrix(imageX, 'offset', [distV,distH]);
        media = mean(mean(imageX));
        stats = graycoprops(cocurrenceM);
        % Matriz con vectores de caracteristicas
        vectorP = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
        F(line,:) = vectorP;
        L(line,:) = 'D6.BMP ';
        line = line + 1;
    end
end
%Extraccion de caracteristicas de la imagen D64
%Feature extraction of the image D64
line = 1;
for x=1:samples
    for y=1:samples
        imageX = imread('D64.BMP');
        imageX = rgb2gray(imageX);
        imageX = imageX(1*x:64*x,1*y:64*y);
        imageX = histeq(imageX, level);
        cocurrenceM = zeros(level);
        %Feature extraction
        cocurrenceM = graycomatrix(imageX, 'offset', [distV,distH]);
        media = mean(mean(imageX));
        stats = graycoprops(cocurrenceM);
        % Matriz con vectores de caracteristicas
        vectorP = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
        F2(line,:) = vectorP;
        L2(line,:) = 'D64.BMP';
        line = line + 1;
    end
end
%Extraccion de caracteristicas de la imagen 22.tiff
line = 1;
for x=1:samples
    for y=1:samples
        imageX = imread('22.tiff');
        %imageX = rgb2gray(imageX);
        imageX = imageX(1*x:32*x,1*y:32*y);
        imageX = histeq(imageX, level);
        cocurrenceM = zeros(level);
        %Feature extraction
        cocurrenceM = graycomatrix(imageX, 'offset', [distV,distH]);
        media = mean(mean(imageX));
        stats = graycoprops(cocurrenceM);
        % Matriz con vectores de caracteristicas
        vectorP = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
        F3(line,:) = vectorP;
        L3(line,:) = '22.tiff';
        line = line + 1;
    end
end
%concatenacion de las matrices
Fr = vertcat(F, F2, F3);
Lr = vertcat(L, L2, L3);

NB = fitcnb(Fr, Lr);
%Mdl = fitcdiscr(Fr, Lr);
KNN = ClassificationKNN.fit(Fr, Lr);


%Set de prueba de D6

imgTest = imread('D6.bmp');
imgTest = rgb2gray(imgTest);
[x,y]=size(imgTest);
line=1;
for i=1:10
    for j=1:10
       prueba = imgTest(1*i:(x/10)*i,1*j:(y/10)*j);
       prueba = histeq(prueba,level);
       cocurrenceM = graycomatrix(prueba, 'offset', [distV,distH]);
       media = mean(mean(prueba));
       stats = graycoprops(cocurrenceM);
       vectorF = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
       resultado1KNN(line) = {[predict(KNN,vectorF)]};
       labelsD6(line) = {['D6.BMP ']};
       resultado1NB(line)= {[predict(NB,vectorF)]};
       line=line+1;
    end
end

%Set de prueba de D64

imgTest = imread('D64.bmp');
imgTest = rgb2gray(imgTest);
[x,y]=size(imgTest);
line=1;
for i=1:10
    for j=1:10
       prueba = imgTest(1*i:(x/25)*i,1*j:(y/25)*j);
       prueba = histeq(prueba,level);
       cocurrenceM = graycomatrix(prueba, 'offset', [distV,distH]);
       media = mean(mean(prueba));
       stats = graycoprops(cocurrenceM);
       vectorF = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
       resultado2KNN(line) = {[predict(KNN,vectorF)]};
       labelsD64(line) = {['D64.BMP']};
       resultado2NB(line)= {[predict(NB,vectorF)]};
       line=line+1;
    end
end

truthLabels = horzcat(labelsD6, labelsD64);
outKNN = horzcat(resultado1KNN, resultado2KNN);
outNB = horzcat(resultado1NB, resultado2NB);

%Analisis con classperf
CPKNN = classperf(truthLabels, outKNN)
CPNB = classperf(truthLabels, outNB)