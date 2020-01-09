function [inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Context_inpFun

numConds = 40;

perturbDirs = [pi/2 pi/2*3];
%% General inputs and output
inp = cell(1,numConds);
targ = cell(1,numConds);
splitPoint = numConds/2;
targetFunPassthrough = [];
for cond = 1:numConds
    fixTime = 0;
    moveTime = randi(50)+50;
    preTime = randi(40)+40;
    totalTime = preTime + fixTime + moveTime;
    
    startPoint = [0 0]';
    endPoint = [0 1]';
    
    inp{cond} = zeros(2, totalTime);
  %  inp{cond}(1:2,:) = repmat(endPoint, [1 totalTime]);
    if cond <= splitPoint
        inp{cond}(1,:) = zeros(1, totalTime);  
        thisCond = 1;
    else
        inp{cond}(1,:) = ones(1, totalTime);
        thisCond = 2;
    end
    inp{cond}(2,:) = [ones(1,preTime+fixTime-10) zeros(1,moveTime+10)];

    targ{cond} = [repmat(startPoint, [1 preTime+fixTime]), repmat(endPoint, [1 moveTime])];
    targ{cond}(:,1:end-1) = nan;
    
    targetFunPassthrough(cond).thisCond = thisCond;
    targetFunPassthrough(cond).perturbTrials = randi(2)-1;
    targetFunPassthrough(cond).perturbDir = perturbDirs(randi(2));
    targetFunPassthrough(cond).perturbMag = 0.4;%(rand * 0.5) + 0.1; %0.4
    targetFunPassthrough(cond).perturbDist = 0.1;
    targetFunPassthrough(cond).kinStart = preTime; % Only allow movement
    targetFunPassthrough(cond).pos = startPoint;
    targetFunPassthrough(cond).end = endPoint;
end
fitnessFunInputs = targ;

end