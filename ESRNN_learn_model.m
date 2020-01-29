function [winner, varargout] = ESRNN_learn_model(inp, learningRate, mutationPower, populationSize, fitnessFunInputs, policyInitInputs, varargin)

% net = geneticRNN_learn_model(inp, mutationPower, populationSize, fitnessFunInputs, policyInitInputs, varargin)
%
% This function trains a recurrent neural network using a simple genetic algorithm
% to complete the desired goal.
%
% INPUTS:
%
% inp -- Inputs to the network. Must be present, but can be empty.
%
% mutationPower -- Standard deviation of normally distributed noise to add in each generation
%
% populationSize -- Number of individuals in each generation
%
% truncationSize -- Number of individuals to save for next generation
%
% fitnessFunInputs -- Target information for calculating the fitness
%
% policyInitInputs -- Inputs for the policy initialization function
%
%
% OPTIONAL INPUTS:
%
% mutationPowerDecay -- Natural decay rate of mutation power
%
% mutationPowerDrop -- Decay rate of mutation power when we don't learn anything on a given generation
%
% weightCompression -- Whether or not to compress policy (logical)
%
% weightDecay -- Whether or not to decay policy (logical)
%
% fitnessFun -- function handle for assessing fitness
% Default: @defaultFitnessFunction
%
% policyInitFun -- function handle for initializing the policy
% Default: @geneticRNN_create_model
%
% policyInitInputsOptional -- Optional inputs for the policy initialization function
%
% targetFun -- The handle of a function that uses the firing rates of the
% output units to produce some desired output. Function must follow
% conventions of supplied default function.
% Default: @defaultTargetFunction
%
% targetFunPassthrough -- A user-defined structure that is automatically
% passed through to the targetFun, permitting custom variables to be passed
% Default: []
%
% plotFun -- The handle of a function that plots information about the
% network during the learning process. Function must follow conventions
% of supplied default function.
% Default: @defaultPlottingFunction
%
% evalOpts -- A vector of size 2, specifying how much information should be
% displayed during training (0 - nothing, 1 - text only, 2 - text +
% figures), and how often the network should be evaluated. This vector is
% passed to the plotting function.
% Default: [1 1]
%
%
% OUTPUTS:
%
% winner -- the network structure
%
% errStats -- the structure containing error information from learning
% (optional)
%
%
% Copyright (c) Jonathan A Michaels 2018
% Stanford University
% jonathanamichaels AT gmail DOT com


% Start counting
tic

% Variable output considerations
nout = max(nargout,1)-1;

% Variable input considerations
optargin = size(varargin,2);

targetFun = @defaultTargetFunction; % Default output function (native)
plotFun = @defaultPlottingFunction; % Default plotting function (native)
fitnessFun = @defaultFitnessFunction; % Default fitness function (native)
policyInitFun = @ESRNN_create_model;
policyInitInputsOptional = [];
targetFunPassthrough = []; % Default passthrough to output function
evalOpts = [1 1]; % Default evaluation values [plottingOptions evaluateEveryXIterations]
optimizer = 'SGD';
optimizerParams = [0.001, 0.9 0.999, 1e-8];
batchSize = 1;
maxIters = Inf;
checkpointFreq = 0;

for iVar = 1:2:optargin
    switch varargin{iVar}
        case 'fitnessFun'
            fitnessFun = varargin{iVar+1};
        case 'policyInitFun'
            policyInitFun = varargin{iVar+1};
        case 'policyInitInputsOptional'
            policyInitInputsOptional = varargin{iVar+1};
            
        case 'targetFun'
            targetFun = varargin{iVar+1};
        case 'targetFunPassthrough'
            targetFunPassthrough = varargin{iVar+1};
            
        case 'optimizer'
            optimizer = varargin{iVar+1};
        case 'optimizerParams'
            optimizerParams = varargin{iVar+1};
            
        case 'batchSize'
            batchSize = varargin{iVar+1};
        case 'maxIters'
            maxIters = varargin{iVar+1};
            
        case 'plotFun'
            plotFun = varargin{iVar+1};
        case 'evalOpts'
            evalOpts = varargin{iVar+1};
            
        case 'checkpointFreq'
            checkpointFreq = varargin{iVar+1};
        case 'checkpointPath'
            checkpointPath = varargin{iVar+1};
    end
end

errStats.fitness = []; errStats.generation = []; errStats.time = [];% Initialize error statistics
g = 1; % Initialize generation

% Build utility function
d = zeros(1,populationSize);
for k = 1:populationSize
    d(k) = max([0, log(populationSize/2 + 1) - log(k)]);
end
denominator = sum(d);
u = zeros(1,populationSize);
for k = 1:populationSize
    u(k) = (max([0, log(populationSize/2 + 1) - log(k)]) / denominator) - (1/populationSize);
end
u = u / std(u);

% initialize adam
if strcmp(optimizer, 'Adam') || strcmp(optimizer, 'AdaMax')
    m = 0;
    v = 0;
end

if isa(inp, 'function_handle')
    inpFun = inp;
    useFun = true;
else
    useFun = false;
end

decayRate = 0;
fitness = -Inf;

%% Main Program %%
% Runs until tolerated error is met or stop button is pressed
figure(97)
set(gcf, 'Position', [0 50 100 50], 'MenuBar', 'none', 'ToolBar', 'none', 'Name', 'Stop', 'NumberTitle', 'off')
UIButton = uicontrol('Style', 'togglebutton', 'String', 'STOP', 'Position', [0 0 100 50], 'FontSize', 25);
while UIButton.Value == 0 && g < maxIters && mean(fitness) ~= 0
    tic
    
    if useFun
       [inp, fitnessFunInputs, targetFunPassthrough] = inpFun();
    end
    
    %% Generate random seeds
    tempSeeds = randsample(1e8, populationSize/2)';
    tempSeeds2 = [tempSeeds; tempSeeds];
    theseSeeds = tempSeeds2(:);
    
    %% Generate initial policy
    if g == 1
        initSeeds = randsample(1e8, populationSize/10);
        fitness = zeros(length(inp),populationSize/10);
        parfor i = 1:populationSize/10
            % Hack the random number generator
            stream = RandStream('mrg32k3a');
            RandStream.setGlobalStream(stream);
            stream.Substream = i;
            rng(initSeeds(i)) % set the precious seed
            Pnet = policyInitFun(policyInitInputs, policyInitInputsOptional);
            % Run model
            [Z0, Z1, ~, ~, kin] = ESRNN_run_model(Pnet, inp, 'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough);
            % Assess fitness
            fitness(:,i) = fitnessFun(Z0, Z1, kin, fitnessFunInputs);
        end
        [~, sortInd] = sort(mean(fitness,1), 'descend');
        % Hack the random number generator
        stream = RandStream('mrg32k3a');
        RandStream.setGlobalStream(stream);
        stream.Substream = sortInd(1);
        rng(initSeeds(sortInd(1))) % set the precious seed
        net = policyInitFun(policyInitInputs, policyInitInputsOptional);
    end
    
    %% Initialize fitness
    fitness = zeros(1,populationSize);
    
%     %% Initialize decay
%     if strcmp(optimizer, 'SGD')
%         decay1 = learningRate^2;
%     elseif strcmp(optimizer, 'Adam')
%         decay1 = optimizerParams(1) * sqrt(1 - optimizerParams(3)^g) / (1 - optimizerParams(2)^g);
%     end
    
    %% Heavy lifting
    modMask = net.modMask;
    parfor i = 1:populationSize
        %% Choose trials for batch
        idx = randsample(length(inp),round(length(inp)*batchSize));
        useInp = inp(idx);
        useFitnessFunInputs = fitnessFunInputs(idx);
        if ~isempty(targetFunPassthrough)
            useTargetFunPassthrough = targetFunPassthrough(idx);
        else
            useTargetFunPassthrough = [];
        end
        % Hack the random number generator
        stream = RandStream('mrg32k3a');
        RandStream.setGlobalStream(stream);
        stream.Substream = i;
        rng(theseSeeds(i)) % set the precious seed
        Pnet = net;
        if mod(i,2) == 0 % Using antithetic sampling
            Pnet.theta = Pnet.theta + ((randn(size(Pnet.theta,1),1) * mutationPower) .* modMask);
        else
            Pnet.theta = Pnet.theta + ((-randn(size(Pnet.theta,1),1) * mutationPower) .* modMask);
        end
        % Run model
        [Z0, Z1, ~, ~, kin] = ESRNN_run_model(Pnet, useInp, 'targetFun', targetFun, 'targetFunPassthrough', useTargetFunPassthrough);
        % Assess fitness
        fitness(i) = mean(fitnessFun(Z0, Z1, kin, useFitnessFunInputs));
    end
    
    % Sort fitness
    mF = fitness;
    [~, sortInd] = sort(mF, 'descend');
    rank = 1:populationSize;
    rank(sortInd) = rank;
    utility = u(rank);
    
    mF = mF - mean(mF);
    mF = mF / std(mF);
    utility = mF;
    
    theta = zeros(size(net.theta,1),populationSize);
    modMask = net.modMask;
    currentTheta = net.theta;
    parfor i = 1:populationSize
        % Hack the random number generator
        stream = RandStream('mrg32k3a');
        RandStream.setGlobalStream(stream);
        stream.Substream = i;
        rng(theseSeeds(i)) % set the precious seed
        if mod(i,2) == 0
            theta(:,i) = ((randn(size(currentTheta,1),1) * mutationPower) .* modMask * utility(i));
        else
            theta(:,i) = ((-randn(size(currentTheta,1),1) * mutationPower) .* modMask * utility(i));
        end
    end
    theta = sum(theta,2);
    
    grad = -1 / (mutationPower*populationSize) * theta;
    
    if strcmp(optimizer, 'SGD')
        newGrad = learningRate * grad;
    elseif  strcmp(optimizer, 'Adam')
        m = optimizerParams(2) * m + (1 - optimizerParams(2)) * grad;
        v = optimizerParams(3) * v + (1 - optimizerParams(3)) * (grad .* grad);
        mC = m / (1 - optimizerParams(2)^g);
        vC = v / (1 - optimizerParams(3)^g);
        newGrad = (optimizerParams(1) * mC) ./ (sqrt(vC) + optimizerParams(4));
    elseif strcmp(optimizer, 'AdaMax')
        m = optimizerParams(2) * m + (1 - optimizerParams(2)) * grad;
        if v == 0
            v = abs(grad);
        else
            v = max([optimizerParams(3) * v, abs(grad)], [], 2);
        end
        newGrad = (optimizerParams(1) / (1 - optimizerParams(2)^g)) * (m ./ v);
        newGrad(isnan(newGrad)) = 0;
    end
    
    %% Holy update
    net.theta = net.theta - newGrad; 
    
    %d = 0.001 / sqrt(net.N) * decayRate;
    %net.theta(net.theta < -d) = net.theta(net.theta < -d) + d;
    %net.theta(net.theta > d) = net.theta(net.theta > d) - d;
    
    %% "Cleverly" determine the decay rate
    [~,J,~,~,~,~,~] = ESRNN_unpack(net, net.theta);
    e = abs(eig(J));
    e = e(1);
    decay1 = (1 - 1/e) * 0.2;
    if decay1 < 0
        decay1 = 0;
    end
    
    %% Holy decay
    net.theta = net.theta - decay1 * (net.theta .* modMask);
    
    %% Decay learning rates
    learningRate = learningRate * 0.99;
    %optimizerParams(1) = optimizerParams(1) * 0.99;
    decayRate = decayRate * 0.99;
    
    %% Recalculate best network for plotting or output
    % Run model
    [Z0, Z1, R, ~, kin] = ESRNN_run_model(net, inp, 'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough);
    fitness = fitnessFun(Z0, Z1, kin, fitnessFunInputs);
    
    [~, J, ~, ~, ~, ~, ~] = ESRNN_unpack(net, net.theta);
    e = abs(eig(J));
    disp([num2str(e(1)) '  ' num2str(optimizerParams(1))])
    %disp(optimizerParams(1))
    %  disp(decay1)
    
    %% Save stats
    errStats.fitness(:,end+1) = fitness;
    errStats.generation(end+1) = g;
    errStats.time(end+1) = cputime;
    
    %% Populate statistics for plotting function
    plotStats.fitness = fitness';
    plotStats.mutationPower = mutationPower;
    plotStats.generation = g;
    plotStats.bigZ0 = Z0;
    plotStats.bigZ1 = Z1;
    plotStats.bigR = R;
    plotStats.targ = fitnessFunInputs;
    plotStats.kin = kin;
    plotStats.tRun = toc;
    
    %% Run supplied plotting function
    if mod(g, evalOpts(2)) == 0
        plotFun(plotStats, errStats, evalOpts)
    end
    
    if mod(g, checkpointFreq) == 0
        fOut = num2str(mean(fitness));
        fOut(strfind(fOut,'.')) = ',';
        save([checkpointPath '//ESRNN_generation-' num2str(g) '_fitness' fOut '.mat'], 'net', 'inp', ...
            'fitnessFun', 'fitnessFunInputs', 'targetFun', 'targetFunPassthrough', 'errStats')
    end
    
    g = g + 1;
end

%% Output error statistics if required
if ( nout >= 1 )
    varargout{1} = errStats;
end

%% Save hard-earned elite network
winner = net;
if checkpointFreq ~= 0
    fOut = num2str(mean(fitness));
    fOut(strfind(fOut,'.')) = ',';
    save([checkpointPath '//ESRNN_generation-' num2str(g) '_fitness' fOut '.mat'], 'net', 'inp', ...
        'fitnessFun', 'fitnessFunInputs', 'targetFun', 'targetFunPassthrough', 'errStats')
end

%% Default plotting function
    function defaultPlottingFunction(plotStats, errStats, evalOptions)
        if evalOptions(1) >= 0
            disp(['Generation: ' num2str(plotStats.generation) '  Fitness: ' num2str(mean(plotStats.fitness(:,1))) '  Time Required: ' num2str(plotStats.tRun) ' seconds'])
        end
        if evalOptions(1) >= 1
            figure(98)
            set(gcf, 'Name', 'Error', 'NumberTitle', 'off')
            c = lines(size(plotStats.fitness,1));
            for type = 1:size(plotStats.fitness,1)
                h1(type) = plot(plotStats.generation, plotStats.fitness(type,1), '.', 'MarkerSize', 20, 'Color', c(type,:));
                hold on
            end
            plot(plotStats.generation, mean(plotStats.fitness(:,1),1), '.', 'MarkerSize', 40, 'Color', [0 0 0]);
            set(gca, 'XLim', [1 plotStats.generation+0.1])
            xlabel('Generation')
            ylabel('Fitness')
        end
        if evalOptions(1) >= 2
            figure(99)
            set(gcf, 'Name', 'Output and Neural Activity', 'NumberTitle', 'off')
            clf
            subplot(4,1,1)
            hold on
            c = lines(length(plotStats.bigZ1));
            for condCount = 1:length(plotStats.bigZ1)
                h2(condCount,:) = plot(plotStats.bigZ1{condCount}', 'Color', c(condCount,:));
                h3(condCount,:) = plot(plotStats.targ{condCount}', '.', 'MarkerSize', 8, 'Color', c(condCount,:));
            end
            legend([h2(1,1) h3(1,1)], 'Network Output', 'Target Output', 'Location', 'SouthWest')
            xlabel('Time Steps')
            ylabel('Output')
            set(gca, 'XLim', [1 size(plotStats.bigZ1{1},2)])
            for n = 1:3
                subplot(4,1,n+1)
                hold on
                for condCount = 1:length(plotStats.bigR)
                    plot(plotStats.bigR{condCount}(n,:)', 'Color', c(condCount,:))
                end
                xlabel('Time Steps')
                ylabel(['Firing Rate (Neuron ' num2str(n) ')'])
                set(gca, 'XLim', [1 size(plotStats.bigR{1},2)])
            end
        end
        drawnow
    end

%% Default fitness function
    function fitness = defaultFitnessFunction(Z0, Z1, kin, targ)
        fitness = zeros(1,length(Z1));
        for cond = 1:length(Z1)
            ind = ~isnan(targ{cond});
            useZ1 = Z1{cond}(ind);
            useF = targ{cond}(ind);
            
            err(1) = sqrt(mean((useZ1(:)-useF(:)).^2));
            %err(2) = sqrt(mean((Z0{cond}(:).^2)));
            err(2) = 0; % by default we won't penalize output cost, since this is task specific
            fitness(cond) = -sum(err);
        end
    end

%% Default output function
    function [z, targetFeedforward] = defaultTargetFunction(~, r, ~, targetFeedforward)
        z = r; % Just passes firing rate
        targetFeedforward.Feedback = z;
    end
end