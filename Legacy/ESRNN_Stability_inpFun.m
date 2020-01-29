function [inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Perturb_inpFun

numConds = 50;

%% General inputs and output
inp = cell(1,numConds);
targ = cell(1,numConds);
targetFunPassthrough = [];
for cond = 1:numConds
    fixTime = randi(40)+40;
    preTime = 30;
    moveTime = 0;
    totalTime = preTime + fixTime + moveTime;
    
    startPoint = (rand(2,1)-0.5)*2;
    endPoint = (rand(2,1)-0.5)*2;
    
    d = sqrt(sum((endPoint - startPoint).^2));
    while d < 0.2
        startPoint = (rand(2,1)-0.5)*2;
        endPoint = (rand(2,1)-0.5)*2;
        d = sqrt(sum((endPoint - startPoint).^2));
    end
    
    inp{cond} = zeros(5, totalTime);
    inp{cond}(1:2,:) = repmat(endPoint, [1 totalTime]);
    inp{cond}(3:4,:) = [repmat(startPoint, [1 preTime+fixTime]) zeros(2, moveTime)];
    inp{cond}(5,:) = [ones(1,preTime+fixTime), zeros(1,moveTime)];
    
    targ{cond} = [repmat(startPoint, [1 preTime+fixTime]), repmat(endPoint, [1 moveTime])];
    targ{cond}(:,[preTime+fixTime+1:end-1]) = nan;
    
    targetFunPassthrough(cond).perturbTrials = 0;
    targetFunPassthrough(cond).perturbDir = rand * pi * 2;
    targetFunPassthrough(cond).perturbMag = rand * 0.3;
    targetFunPassthrough(cond).perturbDist = (rand * 0.2) + 0.1;
    targetFunPassthrough(cond).kinStart = preTime;
    targetFunPassthrough(cond).goTime = preTime+fixTime;
    targetFunPassthrough(cond).pos = startPoint;
    targetFunPassthrough(cond).end = endPoint;
end
fitnessFunInputs = targ;
end