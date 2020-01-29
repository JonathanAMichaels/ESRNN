function [inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Perturb_inpFun

numConds = 100;

%% General inputs and output
inp = cell(1,numConds);
targ = cell(1,numConds);
targetFunPassthrough = [];
for cond = 1:numConds
    fixTime = 0;
    preTime = randi(40)+40;
    moveTime = randi(50)+50;
    totalTime = preTime + fixTime + moveTime;
    
    startPoint = (rand(2,1)-0.5)*2;
    endPoint = (rand(2,1)-0.5)*2;
    
    d = sqrt(sum((endPoint - startPoint).^2));
    while d < 0.2
        startPoint = (rand(2,1)-0.5)*2;
        endPoint = (rand(2,1)-0.5)*2;
        d = sqrt(sum((endPoint - startPoint).^2));
    end
    
    inp{cond} = zeros(3, totalTime);
    inp{cond}(1:2,:) = repmat(endPoint, [1 totalTime]);
    inp{cond}(3,:) = [ones(1,preTime-10), zeros(1,moveTime+10)];
    
    targ{cond} = [repmat(startPoint, [1 preTime+fixTime]), repmat(endPoint, [1 moveTime])];
    targ{cond}(:,2:end-1) = nan;
    
    targetFunPassthrough(cond).perturbTrials = randi(2)-1;
    targetFunPassthrough(cond).perturbDir = rand * pi * 2;
    targetFunPassthrough(cond).perturbMag = 0.6;%rand * 0.3;
    targetFunPassthrough(cond).perturbDist = (rand * 0.2) + 0.1;
    targetFunPassthrough(cond).kinStart = preTime;
    targetFunPassthrough(cond).pos = startPoint;
    targetFunPassthrough(cond).end = endPoint;
    
    targ{cond}(3,:) = targetFunPassthrough(cond).perturbTrials;
 
end
fitnessFunInputs = targ;
end