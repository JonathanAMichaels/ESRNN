function [inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Perturb_inpFun_COP

numConds = 5;
moveTime = 100;

ang = linspace(0, 2*pi - 2*pi/numConds, numConds);

%% General inputs and output
allEndPoint = [sin(ang); cos(ang)];
allStartPoint = zeros(2,numConds);

perturbDirs = ang;

repeats = length(perturbDirs);
%% General inputs and output
inp = cell(1,numConds*(repeats+1));
targ = cell(1,numConds*(repeats+1));
condCount = 1;
for rep = 1:repeats+1
    for cond = 1:size(allStartPoint,2)
        preTime = 30;
        fixTime = 40;
        totalTime = preTime + fixTime + moveTime;

        inp{condCount} = zeros(5, totalTime);
        inp{condCount}(1:2,:) = repmat(allEndPoint(:,cond), [1 totalTime]);
        inp{cond}(3:4,:) = [repmat(allStartPoint(:,cond), [1 preTime+fixTime]) zeros(2, moveTime)];
        inp{condCount}(5,:) = [ones(1,preTime+fixTime)*2 zeros(1,moveTime)];

        targ{condCount} = [repmat(allStartPoint(:,cond), [1 preTime+fixTime]), repmat(allEndPoint(:,cond), [1 moveTime])];
        targ{condCount}(:,[preTime+fixTime+1:end-1]) = nan;
        
        if rep <= length(ang)
            targetFunPassthrough(condCount).perturbDir = ang(rep);
            targetFunPassthrough(condCount).perturbTrials = 1;
        else
            targetFunPassthrough(condCount).perturbDir = nan;
            targetFunPassthrough(condCount).perturbTrials = 0;
        end
       
      %  targetFunPassthrough(condCount).thisCond = 1;
        targetFunPassthrough(condCount).perturbMag = 0.3;
        targetFunPassthrough(condCount).perturbDist = 0.1;
        targetFunPassthrough(condCount).kinStart = preTime; % Only allow movement
        targetFunPassthrough(condCount).pos = allStartPoint(:,cond);
        targetFunPassthrough(condCount).end = allEndPoint(:,cond);
        targetFunPassthrough(condCount).goTime = preTime+fixTime;
        
        condCount = condCount + 1;
    end
end

fitnessFunInputs = targ;

end