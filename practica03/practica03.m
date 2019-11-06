%%
% Universidad Nacional Autonoma de México
% Facultad de Ingeniería
% Reconocimiento de Patrones
% 2020-1
% Practica: Reconocimiento de Texturas
% Integrantes:
% Aguilar Enriquez Paul Sebastian
% Padilla Herrera Carlos Ignacio
% Ramirez Ancona Simon Eduardo
% Toolbox:
% Machine learning, Bionfirmatics

%% Cleaning the environment
clear var;
clear globalvar;
close all;
clc;

%% Define the number of samples 
samples = 5;
level = 10;

%% Perfoming analysis using distance and degrees
verticalDistance = 8;
horizontalDistance = 8;


%% Image feature extraction (using D6 image)
line = 1;
for x=1:samples
    for y=1:samples
        imageX = imread('D08.bmp'); %
        imageX = rgb2gray(imageX);  % RGB TO GRAY
        imageX = imageX(1*x:64*x,1*y:64*y); % 64*64
        imageX = histeq(imageX, level); %enhance contrast using histogram equalization
        cocurrenceM = zeros(level); % Co-ocurrence matrix
        
        % Feature extraction
        cocurrenceM = graycomatrix(imageX, 'offset', [verticalDistance,horizontalDistance]);
        media = mean(mean(imageX));
        stats = graycoprops(cocurrenceM);
        
        % Feature vector matrix
        vectorP = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
        F(line,:) = vectorP;
        L(line,:) = 'D08.bmp';
        line = line + 1;
    end
end

%% Feature extraction of the image D48
line = 1;
for x=1:samples
    for y=1:samples
        imageX = imread('D48.bmp');
        imageX = rgb2gray(imageX);
        imageX = imageX(1*x:64*x,1*y:64*y);
        imageX = histeq(imageX, level);
        cocurrenceM = zeros(level);
        %Feature extraction
        cocurrenceM = graycomatrix(imageX, 'offset', [verticalDistance,horizontalDistance]);
        media = mean(mean(imageX));
        stats = graycoprops(cocurrenceM);
        % Matrix with feature vectors
        vectorP = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
        F2(line,:) = vectorP;
        L2(line,:) = 'D48.bmp';
        line = line + 1;
    end
end

%% Feature extraction of image D26.bmp
line = 1;
for x=1:samples
    for y=1:samples
        imageX = imread('D26.bmp'); %% change this line asap
        imageX = rgb2gray(imageX);
        imageX = imageX(1*x:32*x,1*y:32*y);
        imageX = histeq(imageX, level);
        cocurrenceM = zeros(level);
        %Feature extraction
        cocurrenceM = graycomatrix(imageX, 'offset', [verticalDistance,horizontalDistance]);
        media = mean(mean(imageX));
        stats = graycoprops(cocurrenceM);
        % Matriz con vectores de caracteristicas
        vectorP = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
        F3(line,:) = vectorP;
        L3(line,:) = 'D26.bmp';
        line = line + 1;
    end
end

%Matrix concatenation
Fr = vertcat(F, F2, F3);
Lr = vertcat(L, L2, L3);


NB = fitcnb(Fr, Lr); % Naive Bayes
%Mdl = fitcdiscr(Fr, Lr); 
KNN = ClassificationKNN.fit(Fr, Lr); % K nearest neighbor


%% Test set using D8.BMP

testImage = imread('D08.bmp'); 
testImage = rgb2gray(testImage);
[x,y]=size(testImage);
line=1;
for i=1:10
    for j=1:10
       test = testImage(1*i:(x/10)*i,1*j:(y/10)*j);
       test = histeq(test,level);
       cocurrenceM = graycomatrix(test, 'offset', [verticalDistance,horizontalDistance]);
       media = mean(mean(test));
       stats = graycoprops(cocurrenceM);
       vectorF = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
       resultado1KNN(line) = {[predict(KNN,vectorF)]};
       labelsD6(line) = {['D08.BMP ']};
       resultado1NB(line)= {[predict(NB,vectorF)]};
       line=line+1;
    end
end

% Test set of D48

testImage = imread('D48.bmp');
testImage = rgb2gray(testImage);
[x,y]=size(testImage);

line=1;
for i=1:10
    for j=1:10
       test = testImage(1*i:(x/25)*i,1*j:(y/25)*j);
       test = histeq(test,level);
       cocurrenceM = graycomatrix(test, 'offset', [verticalDistance,horizontalDistance]);
       media = mean(mean(test));
       stats = graycoprops(cocurrenceM);
       vectorF = [media stats.Contrast stats.Correlation stats.Energy stats.Homogeneity];
       resultado2KNN(line) = {[predict(KNN,vectorF)]};
       labelsD48(line) = {['D48.bmp']};
       resultado2NB(line)= {[predict(NB,vectorF)]};
       line=line+1;
    end
end

trueLabels = horzcat(labelsD6, labelsD48);
outputfromKNN = horzcat(resultado1KNN, resultado2KNN);
outputNaiveBayes = horzcat(resultado1NB, resultado2NB);

%% Analysis with classperf from bionformatics toolbox. 
CPKNN = classperf(trueLabels, outputfromKNN)
CPNB = classperf(trueLabels, outputNaiveBayes)