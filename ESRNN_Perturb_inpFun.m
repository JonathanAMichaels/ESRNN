function [inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Perturb_inpFun

numConds = 10;
moveTimePer = 30;
reps = 5;
%% General inputs and output
inp = cell(1,numConds);
targ = cell(1,numConds);
targetFunPassthrough = [];
for cond = 1:numConds
    targetPoints = (rand(2,reps+1)-0.5)*2;
    
    for i = 2:reps
        d = sqrt(sum((targetPoints(:,i) - targetPoints(:,i-1)).^2));
        while d < 0.2
            targetPoints(:,i) = (rand(2,1)-0.5)*2;
            d = sqrt(sum((targetPoints(:,i) - targetPoints(:,i-1)).^2));
        end
    end
    inp{cond} = zeros(2, moveTimePer*reps);
    for i = 1:reps
        inp{cond}(1:2,(i-1)*moveTimePer+1 : i*moveTimePer) = repmat(targetPoints(:,i+1), [1 moveTimePer]);
    end
    targ{cond} = inp{cond};
    targ{cond}(:,1) = targetPoints(:,1);
    
    targetFunPassthrough(cond).perturbTrials = 0;%randi(2)-1;
    targetFunPassthrough(cond).perturbDir = rand * pi * 2;
    targetFunPassthrough(cond).perturbMag = 0;%rand * 0.3;
    targetFunPassthrough(cond).perturbDist = (rand * 0.2) + 0.1;
    targetFunPassthrough(cond).kinStart = 1;
    targetFunPassthrough(cond).goTime = 1;
    targetFunPassthrough(cond).pos = targetPoints(:,1);
    targetFunPassthrough(cond).end = targetPoints(:,end);
    
end
fitnessFunInputs = targ;
end