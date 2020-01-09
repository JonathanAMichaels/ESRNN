% geneticRNN_Example_DNMS
%
% This function illustrates an example of a simple genetic learning algorithm
% in a recurrent neural network to complete a delayed nonmatch-to-sample
% task.
%
%
% Copyright (c) Jonathan A Michaels 2018
% Stanford University
% jonathanamichaels AT gmail DOT com


clear
close all

%% Generate inputs and outputs
inp = cell(1,4);
targ = cell(1,4);
level = 1;
cue1Time = 1:20;
cue2Time = 41:60;
totalTime = 80;
checkTime = 80;
target1 = 1;
target2 = -1;
for type = 1:4
    inp{type} = zeros(2, totalTime);
    if type == 1
        inp{type}(1, [cue1Time cue2Time]) = level;
        targ{type} = [nan(1, checkTime(1)-1) ones(1, totalTime-checkTime(1)+1)]*target1;
    elseif type == 2
        inp{type}(2, [cue1Time cue2Time]) = level;
        targ{type} = [nan(1, checkTime(1)-1) ones(1, totalTime-checkTime(1)+1)]*target1;
    elseif type == 3
        inp{type}(1, cue1Time) = level;
        inp{type}(2, cue2Time) = level;
        targ{type} = [nan(1, checkTime(1)-1) ones(1, totalTime-checkTime(1)+1)]*target2;
    elseif type == 4
        inp{type}(2, cue1Time) = level;
        inp{type}(1, cue2Time) = level;
        targ{type} = [nan(1, checkTime(1)-1) ones(1, totalTime-checkTime(1)+1)]*target2;
    end
end
% In the delayed nonmatch-to-sample task the network receives two temporally
% separated inputs. Each input lasts 200ms and there is a 200ms gap between them.
% The goal of the task is to respond with one value if the inputs were
% identical, and a different value if they were not. This response must be
% independent of the order of the signals and therefore requires the
% network to remember the first input!

%% Initialize network parameters
N = 100; % Number of neurons
B = size(targ{1},1); % Outputs
I = size(inp{1},1); % Inputs
F = B;
p = 1; % Sparsity
g = 1.5; % Spectral scaling
dt = 10; % Time step
tau = 50; % Time constant

%% Policy initialization parameters
policyInitInputs = {N, B, I, F, p, g, dt, tau};
policyInitInputsOptional = {''};

%% Initialize learning parameters
mutationPower = 2e-2; % Standard deviation of normally distributed noise to add in each generation
populationSize = 5000; % Number of individuals in each generation
batchSize = 1;
learningRate = 1e-1;
optimizerParams = [0.1, 0.9, 0.999, 1e-1];
%optimizerParams = [0.01, 0.9, 0.999];
fitnessFunInputs = targ; % Target data for fitness calculation
evalOpts = [2 1]; % Plotting level and frequency of evaluation

%% Train network
% This step should take less than 5 minutes on a 16 core machine.
% Should be stopped at the desired time by pressing the STOP button and waiting for 1 iteration.
% Look inside to see information about the many optional parameters.
[net, learnStats] = ESRNN_learn_model(inp, learningRate, mutationPower, populationSize, fitnessFunInputs, policyInitInputs, ...
    'evalOpts', evalOpts, ...
    'policyInitInputsOptional', policyInitInputsOptional, ...
    'optimizer', 'Adam', 'optimizerParams', optimizerParams, ...
    'batchSize', batchSize);

% run model
[Z0, Z1, R, X, kin] = ESRNN_run_model(net, inp);

[wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, net.theta);