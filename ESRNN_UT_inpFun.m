function [inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Context_inpFun

numConds = 20;

ang = [-pi/8 pi/8];
endPoints = [sin(ang); cos(ang)];
moveTimes = [30 60];

%% General inputs and output
inp = cell(1,numConds);
targ = cell(1,numConds);
targetFunPassthrough = [];
for cond = 1:numConds
    fixTime = 0;
    speedBlock = randi(2);
    moveTime = moveTimes(speedBlock);
    preTime = randi(40)+40;
    totalTime = preTime + fixTime + moveTime;
    
    startPoint = [0 0]';
    
    inp{cond} = zeros(5, totalTime);
    whichKind = randi(2);
    if whichKind == 1 % single targ    
        useTarg = randi(2);
        inp{cond}(useTarg,:) = ones(1, totalTime);
        targ{cond} = [repmat(startPoint, [1 preTime+fixTime]), repmat(endPoints(:,useTarg), [1 moveTime])]; 
        turnOff = [];
    else % dual targ
        inp{cond}(1:2,:) = ones(2, totalTime);
        turnOff = randi(2);
        useTarg = abs(turnOff-3);       
        targ{cond} = [repmat(startPoint, [1 preTime+fixTime]), repmat(endPoints(:,useTarg), [1 moveTime])];
    end
    if speedBlock == 1
        inp{cond}(3,:) = ones(1, totalTime);
    elseif speedBlock == 2
        inp{cond}(4,:) = ones(1, totalTime);
    end
    inp{cond}(5,:) = [ones(1,preTime+fixTime-10) zeros(1,moveTime+10)];

    targ{cond}(:,1:end-1) = nan;
    
    
    targetFunPassthrough(cond).thisCond = whichKind;
    targetFunPassthrough(cond).turnOff = turnOff;
    targetFunPassthrough(cond).perturbTrials = 0;
    targetFunPassthrough(cond).perturbDir = 0;
    targetFunPassthrough(cond).perturbMag = 0;
    targetFunPassthrough(cond).perturbDist = 0;
    targetFunPassthrough(cond).kinStart = preTime; % Only allow movement
    targetFunPassthrough(cond).pos = startPoint;
    targetFunPassthrough(cond).end = endPoints(:,useTarg);
end
fitnessFunInputs = targ;

end