% clasificacion with a 2 input perceptron
clear 
clc

P = [-0.5 -0.5 +0.3 -0.1; ...
     -0.5 +0.5 -0.5 +1.0];
T = [1 1 0 0];

plotpv (P,T);

net = newp([-1 1; -1 1],1);
plotpv( P,T);
plotpc(net.IW{1},net.b{1});
net.adaptParam.passes = 3;
net = adapt (net,P,T);
plotpc(net.IW{1},net.b{1});

p =[0.7; 1.2];
a = sim(net,p);
plotpv(p,a);
point = findobj (gca,'type','line');
set(point, 'Color', 'red');

hold on;
plotpv(P,T);
plotpc(net,IW{1},net.b{1});
hold off;