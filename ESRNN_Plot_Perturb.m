

count = 1;
for reachDir = 1:4
    for reachDist = 1:2
        for perturbDir = 1:4
            for perturbMag = 1:3
                reachDirVector(count) = reachDir;
                reachDistVector(count) = reachDist;
                perturbDirVector(count) = perturbDir;
                perturbMagVector(count) = perturbMag;
                count = count + 1;
            end
        end
    end
end

clear firingRates bigR r
for reps = 1:10
    [inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Perturb_inpFun_PM;
    % run model
    [Z0, Z1, R, X, kin] = ESRNN_run_model(net, inp, 'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough);
    for cond = 1:length(inp)
        bigR{cond} = R{cond}(:,end-129:end); 
        r(:,cond,:) = bigR{cond}';
        for i = 1:size(R{cond},1)
            firingRates(i, reachDirVector(cond), reachDistVector(cond), ...
                perturbDirVector(cond), perturbMagVector(cond), ...
                :, reps) = bigR{cond}(i,:);
        end
    end
end

fitness = fitnessFun(Z0, Z1, kin, fitnessFunInputs);

[wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, net.theta);

N = net.N;
clear areaInd
%if strcmp(func2str(policyInitFun), 'ESRNN_Perturb_create_model')
    areaInd{1} = 1:N;
%else
%    areaInd{1} = 1 : round(N*0.45);
%    areaInd{2} = round(N*0.45)+1 : round(N*0.90);
%    areaInd{3} = round(N*0.90)+1 : N;
%end

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

numConds = 1;
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
%cc = [lines(numConds); 0 0 0];
%c = [];
%for i = 1:numConds+1
  %  c(end+1:end+numConds,:) = repmat(cc(i,:), [numConds 1]);
%end
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





allFiringRates = firingRates;

for area = 1:length(areaInd)

firingRates = allFiringRates(areaInd{area},:,:,:,:,:,:);

lambdaReps = 1; % 10 is ideal
classificationReps = 100; % 100 is ideal
shuffleReps = 100; % 100 is ideal


firingRatesAverage = mean(firingRates, 7);
trialNum = zeros(size(firingRatesAverage)) + 10;

combinedParams = {{1, [1 5]}, {2, [2 5]}, {3, [3 5]}, ...
    {4, [4 5]}, {5}, {[1 2], [1 2 5]}, {[1 3], [1 3 5]}, ...
    {[1 4], [1 4 5]}, {[2 3], [2 3 5]}, {[2 4], [2 4 5]}, ...
    {[3 4], [3 4 5]}};
margNames = {'ReachDir', 'ReachDist', 'PerturbDir', ...
    'PerturbMag', 'Condition-independent', 'ReachDir/ReachDist', 'ReachDir/PerturbDir', ...
    'ReachDir/PerturbMag', 'ReachDist/PerturbDir', 'ReachDist/PerturbMag', ...
    'PerturbDist/PerturbMag'};
margColours = lines(length(margNames));%[0 0 0.7; 0 0.7 0; 0.7 0 0; 0.7 0.7 0.7];

% Time events of interest (e.g. stimulus onset/offset, cues etc.)
timeEvents = [30];
time = 1:size(firingRatesAverage,6);


optimalLambda = dpca_optimizeLambda(firingRatesAverage, firingRates, trialNum, ...
    'combinedParams', combinedParams, ...
    'numRep', lambdaReps, ...
    'lambdas', 1e-7 * 1.5.^[0:25], ...
    'numComps', 20); % increase this number to ~10 for better accuracy



[W,V,whichMarg] = dpca(firingRatesAverage, 20, ...
    'combinedParams', combinedParams, ...
    'lambda', optimalLambda, ...
    'timeParameter', 5);

explVar = dpca_explainedVariance(firingRatesAverage, W, V, ...
    'combinedParams', combinedParams);

dpca_plot(firingRatesAverage, W, V, @ESRNN_dpca_PlotFun, ...
    'explainedVar', explVar, ...
    'marginalizationNames', margNames, ...
    'marginalizationColours', margColours, ...
    'whichMarg', whichMarg,                 ...
    'time', time,                        ...
    'timeEvents', timeEvents,               ...
    'timeMarginalization', 5, ...
    'legendSubplot', [], ...
    'numCompToShow', 20)
end




X = firingRatesAverage(:,:)';
Xcen = bsxfun(@minus, X, mean(X));
dataDim = size(firingRatesAverage);
Z = Xcen * W;
data = reshape(Z', [size(Z,2) dataDim(2:end)]);
dims = size(data);
data = permute(data, [length(dims) 1:length(dims)-1]);
data = permute(data, [1 2 6 5 4 3]);
dData = reshape(data, size(data,1), size(data,2), []);

