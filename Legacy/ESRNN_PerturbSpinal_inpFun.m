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
    endPoint = startPoint;
    
    inp{cond} = zeros(2, totalTime);
    inp{cond}(1:2,:) = repmat(endPoint, [1 totalTime]);

    targ{cond} = [repmat(startPoint, [1 preTime+fixTime]), repmat(endPoint, [1 moveTime])];
    targ{cond}(:,2:end-1) = nan;
    
    targetFunPassthrough(cond).perturbTrials = 0;
    targetFunPassthrough(cond).perturbDir = 0;
    targetFunPassthrough(cond).perturbMag = 0;
    targetFunPassthrough(cond).perturbDist = 0;
    targetFunPassthrough(cond).kinStart = preTime;
    targetFunPassthrough(cond).pos = startPoint;
    targetFunPassthrough(cond).end = endPoint;
end
fitnessFunInputs = targ;
end