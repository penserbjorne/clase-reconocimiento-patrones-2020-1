%% FEEDFORWARD NEURAL NETWORK
%% To create a feedforward backpropagation network  we can use newff

%% Syntax

  %%  net = newff(PR,[S1 S2... SN1],{ TF1 TF2... TFN1}, BTF, BLF, PF)
 %%       Description
        
 %%           NEWFF(PR,[S1 S2...SN1],{TF1 TF2..)
            
%% Consider this set of data:
% Vector
p = [-6,4;-4,-3;-4,0;-4,2;-2,1; 1,-3;1,2; 2,0;4,2; 4,4; 6,-2;6,4;5,5;4,4;-2,5;-2,7;1,3;2,4;4,6;6,8]
t = [1;1;1;1;1;1;1;1;1;1;1;1;0;0;0;0;0;0;0;0]

%% define  the feed forward neural network
net = newff([0,0;0,0;0 1],[3 1],{hardlimit),'train';
%% define the parameters for: epochs goal & learning rate
net.trainParam.epochs =100
net.trainParam.goal = 0.0001
net.trainParam.lr = 0.05
%% output
net = train(net,p',t');
output = net(P);
error =output-t;
perf = perfom(net,out,t)
%% plot 
figure;
subplot(1,2,1)
plotpv(P,T);
subplot(1,2,2);
plotpv(p2,t)
%% training the neural network
%% net = train(net,p,t)
%% net = newff(minmax(p)).[3 1]