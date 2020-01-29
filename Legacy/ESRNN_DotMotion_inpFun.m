function [inp, targ, targetFunPassthrough] = ESRNN_DotMotion_inpFun()

numConds = 50;%100;
targetFunPassthrough = [];
goal = [-1 1];

inp = cell(1,numConds);
targ = cell(1,numConds);
for cond = 1:numConds
    preTime = randi(40)+40;
    cueTime = 50;
    extraTime = 100; %randi(50)+50;
    totalTime = preTime+cueTime+extraTime; % Total trial time
    
 %   CC = [-0.5 0.5];
    coherence = randi(2) - 1.5; %(rand-0.5);
    vis = repmat(coherence, [1 cueTime]);%(rand(1, cueTime)-0.5) + coherence;
    if coherence > 0
        thisCond = 1;
    elseif coherence < 0
        thisCond = 2;
    elseif coherence == 0
        thisCond = randi(2);
    end
    inp{cond} = zeros(2, totalTime);
    inp{cond}(1,:) = [zeros(1,preTime+10), vis, zeros(1,extraTime-10)];
    
    targType = randi(2);
    if targType == 1
        modifier = 1;
        inp{cond}(2,:) = [zeros(1,preTime+10) ones(1,extraTime+cueTime-10)/2];
    else
        modifier = -1;
        inp{cond}(2,:) = [zeros(1,preTime+10) -ones(1,extraTime+cueTime-10)/2];
    end
    
    targ{cond} = [zeros(2,preTime+10), [ones(1,totalTime-preTime-10)*goal(thisCond)*modifier; zeros(1,totalTime-preTime-10)]];
    
    
  %  targ{cond}(:,1:preTime-1) = nan;
   % targ{cond}(:,preTime+1:end-1) = nan;
    
    targetFunPassthrough(cond).kinStart = preTime;
    targetFunPassthrough(cond).pos = [0; 0];
    targetFunPassthrough(cond).end = targ{cond}(:,end);
    targetFunPassthrough(cond).perturbTrials = 0;
    targetFunPassthrough(cond).perturbDir = 0;
    targetFunPassthrough(cond).perturbMag = 0;
    targetFunPassthrough(cond).perturbDist = 0;
end