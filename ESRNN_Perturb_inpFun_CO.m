function [inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Perturb_inpFun_CO

numConds = 8;
moveTime = 40;
ang = linspace(0, 2*pi - (2*pi/numConds), numConds);

%% General inputs and output
inp = cell(1,numConds);
targ = cell(1,numConds);
targetFunPassthrough = [];
for cond = 1:numConds
    totalTime = moveTime;
    
    startPoint = [0 0]';
    endPoint = [sin(ang(cond)) cos(ang(cond))]';
    
    inp{cond}(1:2,:) = repmat(endPoint, [1 totalTime]);
    targ{cond} = repmat(endPoint, [1 moveTime]);
    
    targetFunPassthrough(cond).perturbTrials = 1;
    targetFunPassthrough(cond).perturbDir = pi/2;
    targetFunPassthrough(cond).perturbMag = 0.0;
    targetFunPassthrough(cond).perturbDist = 0.1;
    targetFunPassthrough(cond).kinStart = 1;
    targetFunPassthrough(cond).pos = startPoint;
    targetFunPassthrough(cond).end = endPoint;
    targetFunPassthrough(cond).goTime = 1;
end
fitnessFunInputs = targ;
end