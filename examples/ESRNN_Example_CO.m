% geneticRNN_Example_CO
%
% This function illustrates an example of a simple genetic learning algorithm
% in recurrent neural networks to complete a center-out reaching task.
%
%
% Copyright (c) Jonathan A Michaels 2018
% Stanford University
% jonathanamichaels AT gmail DOT com


clear
close all

numConds = 8; % Number of peripheral targets. Try changing this number to alter the difficulty!
totalTime = 30; % Total trial time
L = [3 3]; % Length of each segment of the arm

%% Populate target function passthrough data
% This is information that the user can define and passthrough to the
% network output function
targetFunPassthrough.L = L;
targetFunPassthrough.kinTimes = 1:totalTime;

%% General inputs and output
inp = cell(1,numConds);
targ = cell(1,numConds);
ang = linspace(0, 2*pi - 2*pi/numConds, numConds);
for cond = 1:numConds
    inp{cond} = zeros(2, totalTime);
    targ{cond} = [ones(totalTime,1)*sin(ang(cond)) ones(totalTime,1)*cos(ang(cond))]';
    inp{cond} = targ{cond};
    targ{cond}(:,1:end-1) = nan;
end
% In the center-out reaching task the network needs to produce the joint angle
% velocities of a two-segment arm to reach to a number of peripheral
% targets spaced along a circle in the 2D plane, based on the desired target
% specified by the input.

%% Initialize network parameters
N = 50; % Number of neurons
B = size(targ{1},1); % Outputs
I = size(inp{1},1); % Inputs
F = 2;
p = 1; % Sparsity
g = 1.0; % Spectral scaling
dt = 10; % Time step
tau = 50; % Time constant
targetFunPassthrough.dt = dt;

%% Policy initialization parameters
policyInitInputs = {N, B, I, F, p, g, dt, tau};
policyInitInputsOptional = {''};

%% Initialize learning parameters
targetFun = @ESRNN_COTargetFun; % handle of custom target function
fitnessFun = @ESRNN_COFitnessFun;
mutationPower = 2e-2; % Standard deviation of normally distributed noise to add in each generation
populationSize = 5000; % Number of individuals in each generation
batchSize = 0.2;
learningRate = 1e-1;
%optimizerParams = [1, 0.9, 0.999, 1e-0]; %1e-2
optimizerParams = [0.01, 0.9, 0.999];
fitnessFunInputs = targ; % Target data for fitness calculation
evalOpts = [2 1]; % Plotting level and frequency of evaluation

%% Train network
% This step should take less than 5 minutes on a 16 core machine.
% Should be stopped at the desired time by pressing the STOP button and waiting for 1 iteration.
% Look inside to see information about the many optional parameters.
[net, learnStats] = ESRNN_learn_model(inp, learningRate, mutationPower, populationSize, fitnessFunInputs, policyInitInputs, ...
    'evalOpts', evalOpts, ...
    'policyInitInputsOptional', policyInitInputsOptional, ...
    'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough, ...
    'fitnessFun', fitnessFun, ...
    'optimizer', 'AdaMax', 'optimizerParams', optimizerParams, ...
    'batchSize', batchSize, 'maxIters', 75);

% run model
[Z0, Z1, R, X, kin] = ESRNN_run_model(net, inp, 'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough);

[wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, net.theta);

%% Plot center-out reaching results
c = lines(length(inp));
figure(1)
clf
for cond = 1:length(inp)
    h(cond) = filledCircle([targ{cond}(1,end) targ{cond}(2,end)], 0.1, 100, [0.9 0.9 0.9]);
    h(cond).EdgeColor = c(cond,:);
    hold on
end
for cond = 1:length(inp)
    plot(Z1{cond}(1,:), Z1{cond}(2,:), 'Color', c(cond,:), 'Linewidth', 2)
end
axis([-1.3 1.3 -1.3 1.3])
axis square

%% Play short movie showing trained movements for all directions
figure(2)
set(gcf, 'Color', 'white')
for cond = 1:length(inp)
    for t = 1:length(targetFunPassthrough.kinTimes)-1
        clf
        for cond2 = 1:length(inp)
            h(cond2) = filledCircle([targ{cond2}(1,end) targ{cond2}(2,end)], 0.1, 100, [0.9 0.9 0.9]);
            h(cond2).EdgeColor = c(cond2,:);
            hold on
        end
        
        line([kin(cond).initvals(1) kin(cond).posL1(t,1)], ...
            [kin(cond).initvals(2) kin(cond).posL1(t,2)], 'LineWidth', 8, 'Color', 'black')
        line([kin(cond).posL1(t,1) Z1{cond}(1,targetFunPassthrough.kinTimes(t))], ...
            [kin(cond).posL1(t,2) Z1{cond}(2,targetFunPassthrough.kinTimes(t))], 'LineWidth', 8, 'Color', 'black')
        
        axis([-1.2 4.5 kin(cond).initvals(2) 1.2])
        axis off
        drawnow
        pause(0.02)
    end
    pause(0.5)
end