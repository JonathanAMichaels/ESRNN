function [inp,targ, targetFunPassthrough, allCoherence, correctChoice] = ESRNN_DotMotion_inpFun()

numConds = 1000;
targetFunPassthrough = [];
goal = [-1 1];

inp = cell(1,numConds);
targ = cell(1,numConds);
for cond = 1:numConds
    preTime = randi(40)+40;
    cueTime = 50;
    extraTime = 150;
    totalTime = preTime+cueTime+extraTime; % Total trial time
    
    coherence = (rand-0.5) * 0.3;
    allCoherence(cond) = coherence;
    vis = (rand(1, cueTime)-0.5) + coherence;
    if coherence > 0
        thisCond = 1;
    elseif coherence < 0
        thisCond = 2;
    elseif coherence == 0
        thisCond = randi(2);
    end
    inp{cond} = [zeros(1,preTime+10), vis, zeros(1,extraTime-10)];
    targ{cond} = [zeros(2,preTime+10), [ones(1,totalTime-preTime-10)*goal(thisCond); zeros(1,totalTime-preTime-10)]];
    %targ{cond}(:,2:end-1) = nan;
    
    correctChoice(cond) = thisCond;
    
    targetFunPassthrough(cond).kinStart = preTime;
    targetFunPassthrough(cond).pos = [0; 0];
    targetFunPassthrough(cond).end = targ{cond}(:,end);
    targetFunPassthrough(cond).perturbTrials = 0;
    targetFunPassthrough(cond).perturbDir = 0;
    targetFunPassthrough(cond).perturbMag = 0;
    targetFunPassthrough(cond).perturbDist = 0;
end