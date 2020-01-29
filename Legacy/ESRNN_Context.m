clear
close all

local = true;
evalOpts = [2 1]; % Plotting level and frequency of evaluation
try
    if ispc
        baseDir = 'E:\jmichaels\Projects\PC';
    else
    % baseDir = '/Users/jonathanamichaels/Desktop/PC';
        baseDir = '/Users/jonathanamichaels/Desktop/jmichaels/Projects/PC';
    end
    % Select base directory
    cd(baseDir)
    disp('Running local config')
catch
    baseDir = '/tmp/jmichaels/Projects/PC';
    local = false;
    evalOpts = [0 1];
    disp('Running cloud config')
    A = gcp('nocreate');
    if isempty(A)
        parpool(64)
    end
end


%% Initialize network parameters
N = 200; % Number of neurons
B = 2; % Outputs
I = 2; % Inputs
F = 4; % Feedback
p = 1; % Sparsity
g = 1.0; % Spectral scaling
dt = 10; % Time step
tau = 50; % Time constant

%% Policy initialization parameters
policyInitInputs = {N, B, I, F, p, g, dt, tau};
policyInitInputsOptional = {'actFun', 'tanh'};

%% Initialize learning parameters
inpFun = @ESRNN_Context_inpFun;
targetFun = @ESRNN_Context_TargetFun; % handle of custom target function
plotFun = @ESRNN_Context_PlotFun;
policyInitFun = @ESRNN_create_model;
fitnessFun = @ESRNN_Context_FitnessFun;
mutationPower = 2e-2; % Standard deviation of normally distributed noise to add in each generation
populationSize = 10000; % Number of individuals in each generation
batchSize = 0.2;
learningRate = 1e-1;
optimizerParams = [1, 0.9, 0.999, 1e-0];
%optimizerParams = [0.01, 0.9, 0.999];
fitnessFunInputs = [];

%% Train network
% This step should take less than 5 minutes on a 16 core machine.
% Should be stopped at the desired time by pressing the STOP button and waiting for 1 iteration.
% Look inside to see information about the many optional parameters.

[net, learnStats] = ESRNN_learn_model(inpFun, learningRate, mutationPower, populationSize, fitnessFunInputs, policyInitInputs, ...
    'evalOpts', evalOpts, ...
    'policyInitFun', policyInitFun, ...
    'policyInitInputsOptional', policyInitInputsOptional, ...
    'targetFun', targetFun, ...
    'plotFun', plotFun, ...
    'fitnessFun', fitnessFun, ...
    'optimizer', 'Adam', 'optimizerParams', optimizerParams, ...
    'batchSize', batchSize, 'maxIters', 3000, ...
    'checkpointPath', baseDir, 'checkpointFreq', 5);


[inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Context_inpFun;
% run model
[Z0, Z1, R, X, kin] = ESRNN_run_model(net, inp, 'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough);
fitness = fitnessFun(Z0, Z1, kin, fitnessFunInputs);
temp = sum(cell2mat({kin.lock})) / length(kin) * 100;
disp(['Success rate: ' num2str(temp)])

N = net.N;
clear areaInd
areaInd{1} = 1 : N;%round(N*0.47);
%areaInd{2} = round(N*0.47)+1 : round(N*0.94);
%areaInd{3} = round(N*0.94)+1 : round(N*0.97);
%areaInd{4} = round(N*0.97)+1 : N;
% 
% N = net.N;
% areaInd{1} = 1 : round(N*0.45);
% areaInd{2} = round(N*0.45)+1 : round(N*0.9);
% areaInd{3} = round(N*0.9)+1 : round(N*0.95);
% areaInd{4} = round(N*0.95)+1 : N;

clear bigR bigZ useTarg useKin bigVel
offset = 79;
for cond = 1:length(inp)
    bigR{cond} = R{cond}(:,end-offset:end);
    bigZ{cond} = Z1{cond}(:,end-offset:end);
    bigZ0{cond} = Z0{cond}(:,end-offset:end);
    bigVel{cond} = kin(cond).vel(end-offset:end,:);
    useTarg{cond} = fitnessFunInputs{cond}(:,end-offset:end);
    useKin(cond) = kin(cond);
    useKin(cond).perturbOnTime = size(bigZ{cond},2) - (length(kin(cond).t) - kin(cond).perturbOnTime);
    useKin(cond).inTarg = size(bigZ{cond},2) - (length(kin(cond).t) - kin(cond).inTarg);
end


[wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, net.theta);

ctemp = flipud(lines(2));
c = [repmat(ctemp(1,:), [length(inp)/2 1]); repmat(ctemp(2,:), [length(inp)/2 1])];
figure(1)
clf
%for cond = 1:length(bigZ)
  %  h(cond) = filledCircle([useTarg{cond}(1,1) useTarg{cond}(2,1)], 0.05, 100, [1 1 1]);
   % hold on
  %  h(cond) = filledCircle([useTarg{cond}(1,end) useTarg{cond}(2,end)], 0.05, 100, [0.9 0.9 0.9]);
  %  h(cond).EdgeColor = c(cond,:);
%end
hold on
rectangle('Pos', [-0.05 0.9750 0.1 0.1])
rectangle('Pos', [-0.8 0.9750 0.2 0.1])
rectangle('Pos', [0.6 0.9750 0.2 0.1])
for cond = 1:length(bigZ)
    plot(bigZ{cond}(1,:), bigZ{cond}(2,:), 'Color', c(cond,:), 'Linewidth', 2)
    if ~isempty(useKin(cond).perturbOnTime)
        plot(bigZ{cond}(1, useKin(cond).perturbOnTime), ...
        bigZ{cond}(2, useKin(cond).perturbOnTime), '.', 'Color', c(cond,:), 'MarkerSize', 30)
    end
end
axis([-1.3 1.3 -1.3 1.7])
axis square

areaTitle = {'M1', 'S1', 'sp-M', 'sp-S'};
clear pc v
for area = 1:length(areaInd)
    catR = [];
    for cond = 1:length(bigR)
        catR = cat(1, catR, bigR{cond}(areaInd{area},:)');
    end
    [coeff, score, latent] = pca(catR);
    v{area} = cumsum(latent) / sum(latent);
    ind = 1;
    for cond = 1:length(bigR)
        pc{cond,area} = score(ind : ind + size(bigR{cond},2) - 1, :);
        ind = ind + size(bigR{cond},2);
    end
end

figure(3)
clf
count = 1;
for area = 1:length(areaInd)
    for dim = 1:4
        subplot(length(areaInd),4,count)
        hold on
        title(['PC ' num2str(dim)])
        for cond = 1:size(pc,1)
            plot(pc{cond,area}(:,dim), 'LineWidth', 2, 'Color', c(cond,:))
            plot(20, pc{cond,area}(20,dim), '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 20)
            plot(useKin(cond).perturbOnTime, pc{cond,area}(useKin(cond).perturbOnTime,dim), '.', 'Color', c(cond,:), 'Markersize', 20)
            plot(useKin(cond).inTarg, pc{cond,area}(useKin(cond).inTarg,dim), '.', 'Color', [0.8 0.8 0.8], 'MarkerSize', 20)
        end
        if dim == 1
            ylabel(areaTitle{area})
        end
        count = count + 1;
    end
end

dimList = 1:3;
figure(4)
clf
for area = 1:length(areaInd)
    subplot(length(areaInd),1,area)
    hold on
    title(areaTitle{area})
    for cond = 1:size(pc,1)
        plot3(pc{cond,area}(:,dimList(1)), pc{cond,area}(:,dimList(2)), pc{cond,area}(:,dimList(3)), 'LineWidth', 2, 'Color', c(cond,:))
        
        plot3(pc{cond,area}(1,dimList(1)), pc{cond,area}(1,dimList(2)), pc{cond,area}(1,dimList(3)), '.', 'Color', [0 0 0], 'MarkerSize', 20)
        plot3(pc{cond,area}(20,dimList(1)), pc{cond,area}(20,dimList(2)), pc{cond,area}(20,dimList(3)), '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 20)
        plot3(pc{cond,area}(useKin(cond).perturbOnTime,dimList(1)), pc{cond,area}(useKin(cond).perturbOnTime,dimList(2)), pc{cond,area}(useKin(cond).perturbOnTime,dimList(3)), '.', 'Color', c(cond,:), 'MarkerSize', 20)
        plot3(pc{cond,area}(useKin(cond).inTarg,dimList(1)), pc{cond,area}(useKin(cond).inTarg,dimList(2)), pc{cond,area}(useKin(cond).inTarg,dimList(3)), '.', 'Color', [0.8 0.8 0.8], 'MarkerSize', 20)
    end
    xlabel('PC 1')
    ylabel('PC 2')
    zlabel('PC 3')
    view(3)
    axis vis3d
end
    
    
    
    