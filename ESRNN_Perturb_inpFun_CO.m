function [inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Perturb_inpFun_CO

numConds = 8;
moveTime = 100;
ang = linspace(0, 2*pi - (2*pi/numConds), numConds);

%% General inputs and output
inp = cell(1,numConds);
targ = cell(1,numConds);
targetFunPassthrough = [];
for cond = 1:numConds
    fixTime = 0;
    preTime = 60;%randi(40)+40;
    totalTime = preTime + fixTime + moveTime;
    
    startPoint = [0 0]';
    endPoint = [sin(ang(cond)) cos(ang(cond))]';
    
    inp{cond} = zeros(2, totalTime);
    inp{cond}(1:2,:) = repmat(endPoint, [1 totalTime]);
    inp{cond}(3,:) = [ones(1,preTime+fixTime-10) zeros(1,moveTime+10)];
    
    targ{cond} = [repmat(startPoint, [1 preTime+fixTime]), repmat(endPoint, [1 moveTime])];
    targ{cond}(:,preTime+fixTime:end-1) = nan;
    
    targetFunPassthrough(cond).perturbTrials = 1;
    targetFunPassthrough(cond).perturbDir = pi/2;
    targetFunPassthrough(cond).perturbMag = 0.0;%0.3;
    targetFunPassthrough(cond).perturbDist = 0.1;
    targetFunPassthrough(cond).kinStart = preTime;
    targetFunPassthrough(cond).pos = startPoint;
    targetFunPassthrough(cond).end = endPoint;
end
fitnessFunInputs = targ;
end