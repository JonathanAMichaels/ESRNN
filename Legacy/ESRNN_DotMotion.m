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


%% General inputs and output

% In the center-out reaching task the network needs to produce the joint angle
% velocities of a two-segment arm to reach to a number of peripheral
% targets spaced along a circle in the 2D plane, based on the desired target
% specified by the input.

%% Initialize network parameters
N = 200; % Number of neurons
B = 2; % Outputs
I = 2; % Inputs
F = 2;
p = 1;% Sparsity
g = 1.0; % Spectral scaling
dt = 10; % Time step
tau = 50; % Time constant

%% Policy initialization parameters
policyInitInputs = {N, B, I, F, p, g, dt, tau};
policyInitInputsOptional = {''};

%% Initialize learning parameters
fitnessFun = @ESRNN_DotMotion_FitnessFun;
targetFun = @ESRNN_DotMotion_TargetFun2;
inpFun = @ESRNN_DotMotion_inpFun;
plotFun = @ESRNN_DotMotion_PlotFun;
policyInitFun = @ESRNN_DotMotion_create_model;
mutationPower = 2e-2; % Standard deviation of normally distributed noise to add in each generation
populationSize = 10000; % Number of individuals in each generation
batchSize = 0.2;
learningRate = 1e-1;
optimizerParams = [1, 0.9, 0.999, 1];
%optimizerParams = [0.01, 0.9, 0.999];
fitnessFunInputs = []; % Target data for fitness calculation
evalOpts = [2 1]; % Plotting level and frequency of evaluation

%% Train network
% This step should take less than 5 minutes on a 16 core machine.
% Should be stopped at the desired time by pressing the STOP button and waiting for 1 iteration.
% Look inside to see information about the many optional parameters.
[net, learnStats] = ESRNN_learn_model(inpFun, learningRate, mutationPower, populationSize, fitnessFunInputs, policyInitInputs, ...
    'evalOpts', evalOpts, ...
    'policyInitFun', policyInitFun, 'policyInitInputsOptional', policyInitInputsOptional, ...
    'targetFun', targetFun, ...
    'fitnessFun', fitnessFun, ...
    'plotFun', plotFun, ...
    'optimizer', 'Adam', 'optimizerParams', optimizerParams, ...
    'batchSize', batchSize, 'maxIters', 5000, ...
    'checkpointPath', 'E:\jmichaels\Projects\DMRNN', 'checkpointFreq', 5);


% run model
[inp, fitnessFunInputs, targetFunPassthrough, coherence, correctChoice] = ESRNN_DotMotion_inpFun2;
[Z0, Z1, R, X, kin] = ESRNN_run_model(net, inp, 'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough);
fitness = fitnessFun(Z0, Z1, kin, fitnessFunInputs);
[wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, net.theta);

coherence = coherence * 2 * 100;

N = net.N;
areaInd{1} = 1 : round(N*0.5);
areaInd{2} = round(N*0.5)+1 : N;

ind = [];
for cond = 1:length(Z1)
    if sqrt(sum((Z1{cond}(:,end) - [-1; 0]).^2)) >= 0.1 && sqrt(sum((Z1{cond}(:,end) - [1; 0]).^2)) >= 0.1
        ind(end+1) = cond;
    end
end
Z1(ind) = [];
Z0(ind) = [];
R(ind) = [];
X(ind) = [];
kin(ind) = [];
inp(ind) = [];
coherence(ind) = [];
correctChoice(ind) = [];

offset = 200;
r = zeros(offset, length(inp), size(R{1},1));
z = zeros(offset, 2, length(inp));
clear RT zMove rCue choice rAll
for cond = 1:length(inp)
    r(:,cond,:) = R{cond}(:,end-(offset-1):end)';
    z(:,:,cond) = Z1{cond}(:,end-(offset-1):end)';
    
    
    temp = (find(sqrt(sum(z(:,:,cond).^2,2)) > 0.1,1)) * 10;
    RT(cond) = temp;
    
    temp = find(sqrt(sum(z(:,:,cond).^2,2)) > 0.1,1);
    zMove(:,:,cond) = z(temp:temp+89,:,cond);
    
    rCue{cond} = squeeze(r(1:temp,cond,:));
    rAll{cond} = squeeze(r(:,cond,:));
    
    dToLeft(cond) = sqrt(sum((z(end,:,cond) - [-1 0]).^2,2));
    dToRight(cond) = sqrt(sum((z(end,:,cond) - [1 0]).^2,2));
    
    if dToLeft(cond) < 0.1
        choice(cond) = -1;
    else
        choice(cond) = 1;
    end
end


step = 5;
bins = -30 : step : 30;
binCenters = bins(1:end-1)+step/2;
clear freq
for i = 1:length(bins)-1
    ind = find(coherence > bins(i) & coherence < bins(i+1));
    freq(i) = mean(choice(ind));
end

figure(1)
clf
plot(binCenters, freq, 'LineWidth', 2)
xlabel('Percent coherence')
ylabel('Probability left/right choice')
box off

c = cool(2);
figure(2)
clf
for i = 1:2
    subplot(2,1,i)
    hold on
    for cond = 1:size(z,3)
        plot(0:10:size(z,1)*10-1, z(:,i,cond), 'Color', [c(correctChoice(cond),:) 0.2])
    end
    xlabel('Time (ms)')
    if i == 1
        ylabel('X-position')
    else
        ylabel('Y-position')
    end
    set(gca, 'YLim', [-1.2 1.2], 'XLim', [1 size(z,1)*10-1])
    box off
end


figure(3)
clf
hold on
plot(abs(coherence), RT, '.')
lsline
box off
xlabel('Absolute coherence')
ylabel('Reaction time (ms)')


% PCA
clear pcR v
for area = 1:length(areaInd)
    rBig = [];
    for cond = 1:length(rCue)
        rBig = cat(1, rBig, rCue{cond}(:,areaInd{area}));
    end
    [coeff, score, latent] = pca(rBig);
    v{area} = cumsum(latent) / sum(latent);
    ind = 1;
    for cond = 1:length(rCue)
        pcR{cond,area} = score(ind : ind + size(rCue{cond},1)-1,:);
        ind = ind + size(rCue{cond},1);
    end
end
cc = cool(size(pcR,1));
[~, ind] = sort(coherence, 'descend');
cc(ind,:) = cc;
figure(4)
clf
count = 1;
for area = 1:2
    for dim = 1:2
        subplot(2,2,count)
        hold on
        title(['PC ' num2str(dim)])
        for cond = 1:length(rCue)
            plot(0:10:size(pcR{cond,area},1)*10-1, pcR{cond,area}(:,dim), 'Color', [cc(cond,:) 0.2])
            plot(size(pcR{cond,area},1)*10-1, pcR{cond,area}(end,dim), '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 5)
        end
        xlabel('Time (ms)')
        if area == 1
            ylabel('Visual area - score (a.u.)')
        else
            ylabel('Motor area - score (a.u.)')
        end
        count = count + 1;
    end
end



cc = cool(size(zMove,3));
[sortedCoh, ind] = sort(coherence, 'descend');
zMoveSorted = zMove(:,:,ind);
zMoveDiff = diff(zMoveSorted,1,1);
figure(5)
clf
subplot(1,2,1)
hold on
for cond = 1:size(zMoveSorted,3)
    plot(0:10:size(zMoveDiff,1)*10-1,zMoveDiff(:,1,cond), 'Color', [cc(cond,:) 0.2])
end
ylabel('Behavioral Output')
xlabel('Time after movement onset (ms)')
subplot(1,2,2)
plot(abs(sortedCoh), max(abs(squeeze(zMoveDiff(:,1,:)))), '.')
corr(abs(sortedCoh)', max(abs(squeeze(zMoveDiff(:,1,:))))')
xlabel('Absolute coherence')
ylabel('Peak Velocity')
box off

path = 'C:\Users\Jonathan\Dropbox\DotMotion\';
print(figure(1), [path 'CRNN-ResponseFreq'], '-dpng')
print(figure(2), [path 'CRNN-BehavioralFullTrial'], '-dpng')
print(figure(3), [path 'CRNN-RT'], '-dpng')
print(figure(4), [path 'CRNN-PCA-BeforeMove'], '-dpng')
print(figure(5), [path 'CRNN-MovementAnalysis'], '-dpng')
