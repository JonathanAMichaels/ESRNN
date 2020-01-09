clear
close all

local = true;
evalOpts = [2 1]; % Plotting level and frequency of evaluation
try
    if ispc
        baseDir = 'E:\jmichaels\Projects\P';
    else
       % baseDir = '/Users/jonathanamichaels/Desktop/P';
         baseDir = '/Users/jonathanamichaels/Desktop/jmichaels/Projects/P';
    end
    % Select base directory
    cd(baseDir)
    disp('Running local config')
catch
    baseDir = '/tmp/jmichaels/Projects/P';
    local = false;
    evalOpts = [0 1];
    disp('Running cloud config')
    A = gcp('nocreate');
    if isempty(A)
        parpool(64)
    end
end


%% Initialize network parameters
N = 400; % Number of neurons
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
inpFun = @ESRNN_Stability_inpFun;
targetFun = @ESRNN_PerturbSpinal_TargetFun; % handle of custom target function
plotFun = @ESRNN_Perturb_PlotFun;
policyInitFun = @ESRNN_PerturbSpinal_create_model;
fitnessFun = @ESRNN_Perturb_FitnessFun;
mutationPower = 2e-2; % Standard deviation of normally distributed noise to add in each generation
populationSize = 10000; % Number of individuals in each generation
batchSize = 0.2;
learningRate = 1e-1;
optimizerParams = [1, 0.9, 0.999, 1];
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
    'batchSize', batchSize, 'maxIters', 30, ...
    'checkpointPath', baseDir, 'checkpointFreq', 5);
save([baseDir '\PerturbSpinal_Best.mat'], 'net', 'learnStats')


inpFun = @ESRNN_Perturb_inpFun;
policyInitFun = @ESRNN_PerturbSpinal_Load;
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


dirVector = repmat(1:5, [1 6]);
perturbDirVector = [ones(1,5) ones(1,5)+1 ones(1,5)+2 ones(1,5)+3 ones(1,5)+4 ones(1,5)+5];

clear firingRates bigR
for reps = 1:10
    
    [inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Perturb_inpFun_COP;
    % run model
    [Z0, Z1, R, X, kin] = ESRNN_run_model(net, inp, 'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough);
    for cond = 1:length(inp)
        bigR{cond} = R{cond}(:,end-129:end);
        for i = 1:size(R{cond},1)
            firingRates(i, dirVector(cond), perturbDirVector(cond), :,reps) = bigR{cond}(i,:);
        end
    end
end

fitness = fitnessFun(Z0, Z1, kin, fitnessFunInputs);

[wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, net.theta);

[wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, net.modMask);

N = net.N;
clear areaInd
if strcmp(func2str(policyInitFun), 'ESRNN_Perturb_create_model')
    areaInd{1} = 1:N;
else
    areaInd{1} = 1 : round(N*0.45);
    areaInd{2} = round(N*0.45)+1 : round(N*0.90);
    areaInd{3} = round(N*0.90)+1 : N;
end

condCount = 1;
clear bigR bigZ useTarg useKin bigZ0 vel
for cond = 1:length(inp)
    bigR{cond} = R{cond}(:,end-129:end);
    bigZ{cond} = Z1{cond}(:,end-129:end);
    bigZ0{cond} = Z0{cond}(:,end-129:end);%kin(cond).FOut(end-129:end,:)';
    vel(:,cond,:) = diff(Z1{cond}(:,end-99:end),1,2)';
    useTarg{cond} = fitnessFunInputs{cond}(:,end-129:end);
    useKin(cond) = kin(cond);
    useKin(cond).perturbOnTime = size(bigZ{cond},2) - (length(kin(cond).t) - kin(cond).perturbOnTime);
    useKin(cond).inTarg = size(bigZ{cond},2) - (length(kin(cond).t) - kin(cond).inTarg);
end
vel = sqrt(sum(vel.^2, 3));

numConds = 5;
splitPoint = 1:numConds:length(inp);
thisCond = 1;
nonPerturb = bigR(end-numConds+1:end);
for cond = 1:length(inp) - numConds
    
    subtractR{cond} = bigR{cond} - nonPerturb{thisCond};
    
    thisCond = thisCond + 1;
    if thisCond > numConds
        thisCond = 1;
    end
end

%% Plot center-out reaching results
c = lines(length(bigZ));
cc = [lines(numConds); 0 0 0];
c = [];
for i = 1:numConds+1
    c(end+1:end+numConds,:) = repmat(cc(i,:), [numConds 1]);
end
figure(1)
clf
for cond = 1:length(bigZ)
    h(cond) = filledCircle([useTarg{cond}(1,1) useTarg{cond}(2,1)], 0.1, 100, [1 1 1]);
    hold on
    h(cond) = filledCircle([useTarg{cond}(1,end) useTarg{cond}(2,end)], 0.1, 100, [0.9 0.9 0.9]);
    h(cond).EdgeColor = c(cond,:);
end
for cond = 1:length(bigZ)
    plot(bigZ{cond}(1,:), bigZ{cond}(2,:), 'Color', c(cond,:), 'Linewidth', 2)
    plot(bigZ{cond}(1, useKin(cond).perturbOnTime), ...
        bigZ{cond}(2, useKin(cond).perturbOnTime), '.', 'Color', c(cond,:), 'MarkerSize', 20)
end
axis([-1.3 1.3 -1.3 1.3])
axis square

figure(5)
clf
for cond = 1:length(bigZ)
    plot(bigZ0{cond}(1,31:end), bigZ0{cond}(2,31:end), 'Color', c(cond,:), 'Linewidth', 2)
    hold on
    plot(bigZ0{cond}(1, useKin(cond).perturbOnTime), ...
        bigZ0{cond}(2, useKin(cond).perturbOnTime), '.', 'Color', c(cond,:), 'MarkerSize', 20)
end

%bigR = subtractR;
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
    for dim = 1:6
        subplot(length(areaInd),6,count)
        hold on
        title(['PC ' num2str(dim)])
        for cond = 1:size(pc,1)
            plot(pc{cond,area}(:,dim), 'LineWidth', 2, 'Color', c(cond,:))
            plot(20, pc{cond,area}(20,dim), '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 20)
            plot(useKin(cond).perturbOnTime, pc{cond,area}(useKin(cond).perturbOnTime,dim), '.', 'Color', c(cond,:), 'Markersize', 20)
            plot(useKin(cond).inTarg, pc{cond,area}(useKin(cond).inTarg,dim), '.', 'Color', [0.8 0.8 0.8], 'MarkerSize', 7)
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

allFiringRates = firingRates;

for area = 1:length(areaInd)

firingRates = allFiringRates(areaInd{area},:,:,:,:);


lambdaReps = 1; % 10 is ideal
classificationReps = 100; % 100 is ideal
shuffleReps = 100; % 100 is ideal


firingRatesAverage = mean(firingRates,5);
trialNum = zeros(size(firingRatesAverage)) + 10;

combinedParams = {{1, [1 3]}, {2, [2 3]}, {3}, {[1 2], [1 2 3]}};
margNames = {'Dir', 'PerturbDir', 'Condition-independent', 'Dir/PerturbDir'};
margColours = [0 0 0.7; 0 0.7 0; 0.7 0 0; 0.7 0.7 0.7];

% Time events of interest (e.g. stimulus onset/offset, cues etc.)
timeEvents = [30];
time = 1:size(firingRatesAverage,4);


optimalLambda = dpca_optimizeLambda(firingRatesAverage, firingRates, trialNum, ...
    'combinedParams', combinedParams, ...
    'numRep', lambdaReps, ...
    'numComps', 8); % increase this number to ~10 for better accuracy



[W,V,whichMarg] = dpca(firingRatesAverage, 8, ...
    'combinedParams', combinedParams, ...
    'lambda', optimalLambda, ...
    'timeParameter', 3);

explVar = dpca_explainedVariance(firingRatesAverage, W, V, ...
    'combinedParams', combinedParams);

dpca_plot(firingRatesAverage, W, V, @ESRNN_dpca_PlotFun, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 3, ...
    'legendSubplot', [], ...
    'numCompToShow', 8)
end

X = firingRatesAverage(:,:)';
Xcen = bsxfun(@minus, X, mean(X));
dataDim = size(firingRatesAverage);
Z = Xcen * W;
data = reshape(Z', [size(Z,2) dataDim(2:end)]);
dims = size(data);
data = permute(data, [length(dims) 1:length(dims)-1]);
data = permute(data, [1 2 4 3]);
dData = reshape(data, size(data,1), size(data,2), []);
