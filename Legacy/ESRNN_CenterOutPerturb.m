clear
close all

local = true;
evalOpts = [2 1]; % Plotting level and frequency of evaluation
try
    if ispc
        baseDir = 'E:\jmichaels\Projects\COP';
    else
       % baseDir = '/Users/jonathanamichaels/Desktop/COP';
        baseDir = '/Users/jonathanamichaels/Desktop/jmichaels/Projects/COP';
    end
    % Select base directory
    cd(baseDir)
    disp('Running local config')
catch
    baseDir = '/tmp/jmichaels/Projects/COP';
    local = false;
    evalOpts = [0 1];
    disp('Running cloud config')
    A = gcp('nocreate');
    if isempty(A)
        parpool(64)
    end
end

numConds = 3; % Number of peripheral targets. Try changing this number to alter the difficulty!
numReaches = 0;
workspace = 1; % in cm
moveTime = 50; % was 40
preTime = 20;

reachInd = 1:numConds+numReaches;
postureInd = length(reachInd)+1:length(reachInd);

ang = linspace(0, 2*pi - 2*pi/numConds, numConds);

rng(1)
randReachStart = 2*(rand(2,numReaches)-0.5)*workspace;
startingPoint = [zeros(2,numConds), randReachStart];

randDirs = rand(1,numReaches)*2*pi;
endPoint = [cat(2, [sin(ang); cos(ang)] * workspace, ...
    randReachStart + [sin(randDirs); cos(randDirs)] * workspace)];

perturbDirs = ang;

repeats = length(perturbDirs);
%% General inputs and output
inp = cell(1,size(startingPoint,2)*(repeats+1));
targ = cell(1,size(startingPoint,2)*(repeats+1));
P = cell(1,size(startingPoint,2)*(repeats+1));
condCount = 1;
for rep = 1:repeats+1
    for cond = 1:size(startingPoint,2)
        totalTime = preTime + moveTime;
        inp{condCount} = zeros(3, totalTime);
        inp{condCount}(1:2,:) = repmat(endPoint(:,cond), [1 totalTime]);
        
        if ismember(cond,postureInd)
            inp{condCount}(3,:) = ones(1,totalTime);
        else
            inp{condCount}(3,:) = [ones(1, preTime-1) zeros(1, moveTime+1)];
        end
        targ{condCount} = [repmat(startingPoint(:,cond), [1 preTime]), repmat(endPoint(:,cond), [1 moveTime])];
        targ{condCount}(:,1:end-1) = nan;
        
        if rep <= length(ang)
            P{condCount} = ang(rep);
        else
            P{condCount} = nan;
        end
        
        condCount = condCount + 1;
    end
end

targetFunPassthrough.L = [3 3];
targetFunPassthrough.perturbDir = P;
targetFunPassthrough.kinStart = preTime; % Only allow movement 
targetFunPassthrough.pos = repmat(startingPoint, [1 (repeats+1)]);
targetFunPassthrough.end = repmat(endPoint, [1 (repeats+1)]);

%% Initialize network parameters
N = 200; % Number of neurons
B = size(targ{1},1); % Outputs
I = size(inp{1},1); % Inputs
F = 4; % Feedback
p = 1; % Sparsity
g = 1.2; % Spectral scaling
dt = 10; % Time step
targetFunPassthrough.dt = dt;
tau = 50; % Time constant

%% Policy initialization parameters
policyInitInputs = {N, B, I, F, p, g, dt, tau};
policyInitInputsOptional = {'actFun', 'tanh'};

%% Initialize learning parameters
targetFun = @ESRNN_CenterOutPerturb_Arm_TargetFun; % handle of custom target function
plotFun = @ESRNN_CenterOutPerturb_PlotFun;
policyInitFun = @ESRNN_CenterOutPerturb_create_model;
fitnessFun = @ESRNN_CenterOutPerturb_FitnessFun;
mutationPower = 2e-2; % Standard deviation of normally distributed noise to add in each generation
populationSize = 100; % Number of individuals in each generation
batchSize = 1;
learningRate = 1e-4;%1e-1;
optimizerParams = [0.1, 0.9, 0.999, 1e-1];
%optimizerParams = [0.01, 0.9, 0.999];
fitnessFunInputs = targ; % Target data for fitness calculation

%% Train network
% This step should take less than 5 minutes on a 16 core machine.
% Should be stopped at the desired time by pressing the STOP button and waiting for 1 iteration.
% Look inside to see information about the many optional parameters.

[net, learnStats] = ESRNN_learn_model(inp, learningRate, mutationPower, populationSize, fitnessFunInputs, policyInitInputs, ...
    'evalOpts', evalOpts, ...
    'policyInitFun', policyInitFun, ...
    'policyInitInputsOptional', policyInitInputsOptional, ...
    'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough, ...
    'plotFun', plotFun, ...
    'fitnessFun', fitnessFun, ...
    'optimizer', 'Adam', 'optimizerParams', optimizerParams, ...
    'batchSize', batchSize, 'maxIters', 3000, ...
    'checkpointPath', baseDir, 'checkpointFreq', 5);

% run model
[Z0, Z1, R, X, kin] = ESRNN_run_model(net, inp, 'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough);
fitness = fitnessFun(Z0, Z1, kin, targ);


N = net.N;
areaInd{1} = 1 : round(N*0.45);
areaInd{2} = round(N*0.45)+1 : round(N*0.9);
areaInd{3} = round(N*0.9)+1 : round(N*0.95);
areaInd{4} = round(N*0.95)+1 : N;

for cond = 1:length(inp)
    r(:,cond,:) = R{cond}';
    z(:,cond,:) = Z0{cond}';
end

[wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, net.theta);

%% Plot center-out reaching results
cTemp = lines(3);
c = [repmat(cTemp(1,:), [3 1]); repmat(cTemp(2,:), [3 1]); repmat(cTemp(3,:), [3 1]); repmat([0 0 0], [3 1])];
figure(1)
clf
for cond = 1:length(inp)
    h(cond) = filledCircle([targ{cond}(1,1) targ{cond}(2,1)], 0.1, 100, [1 1 1]);
    hold on
    h(cond) = filledCircle([targ{cond}(1,end) targ{cond}(2,end)], 0.1, 100, [0.9 0.9 0.9]);
    h(cond).EdgeColor = c(cond,:);
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
    for t = 1:size(kin(cond).t,2)
        clf
        for cond2 = 1:length(inp)
            h(cond2) = filledCircle([targ{cond2}(1,end) targ{cond2}(2,end)], 0.1, 100, [0.9 0.9 0.9]);
            h(cond2).EdgeColor = c(cond2,:);
            hold on
        end
        
        line([kin(cond).initvals(1) kin(cond).posL1(t,1)], ...
            [kin(cond).initvals(2) kin(cond).posL1(t,2)], 'LineWidth', 4, 'Color', 'black')
        line([kin(cond).posL1(t,1) Z1{cond}(1,t)], ...
            [kin(cond).posL1(t,2) Z1{cond}(2,t)], 'LineWidth', 4, 'Color', 'black')
        
        axis([-1.2 4.5 kin(cond).initvals(2) 1.2])
        axis off
        drawnow
    %    pause(0.01)
    end
    pause(0.8)
end

areaTitle = {'M1', 'S1', 'sp-M', 'sp-S'};

for area = 1:length(areaInd)
    thisR = permute(r(:,:,areaInd{area}), [3 1 2]);
    [coeff, score, latent] = pca(thisR(:,:)');
    v{area} = cumsum(latent) / sum(latent);
    pc{area} = permute(reshape(score', [length(areaInd{area}), size(r,1), size(r,2)]), [2 3 1]);
end

figure(3)
clf
count = 1;
for area = 1:length(areaInd)
    for dim = 1:4
    subplot(length(areaInd),4,count)
    hold on
    title(['PC ' num2str(dim)])
    for cond = 1:size(pc{area},2)
        plot(pc{area}(:,cond,dim), 'LineWidth', 2, 'Color', c(cond,:))
    end
    if dim == 1
        ylabel(areaTitle{area})
    end
    count = count + 1;
    end
end

figure(4)
clf
for area = 1:length(areaInd)
    subplot(length(areaInd),1,area)
    hold on
    title(areaTitle{area})
    for cond = 1:size(pc{area},2)
        plot3(pc{area}(:,cond,1), pc{area}(:,cond,2), pc{area}(:,cond,3), 'LineWidth', 2, 'Color', c(cond,:))
    end
    xlabel('PC 1')
    ylabel('PC 2')
    zlabel('PC 3')
    view(3)
    axis vis3d
end
    
    