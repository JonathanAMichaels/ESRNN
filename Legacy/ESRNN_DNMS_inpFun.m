function  [inp, targ, A, net, B] = ESRNN_DNMS_inpFun(net, aaa, bbb, simparams, all_simdata, do_inputs, do_targets);
   
A = true;
B = [];
numConds = 100;

%% General inputs and output
inp = cell(1,numConds);
targ = cell(1,numConds);
targetFunPassthrough = [];
for cond = 1:numConds
    preTime = 30;
    cue1Time = 30;
    memTime = 50;
    cue2Time = 30;
    endTime = 30;
    totalTime = preTime + cue1Time + memTime + cue2Time + endTime;
    
    targ1 = rand(6,1);
    match = randi(2);
    if match == 1
        targ2 = targ1;
        output = 1;
    else
        targ2 = rand(6,1);
        output = -1;
    end
    
    inp{cond} = [zeros(6, preTime) repmat(targ1, [1 cue1Time]) zeros(6,memTime) repmat(targ2, [1 cue2Time]), ...
        zeros(6, endTime)];
    targ{cond} = repmat(output, [1 totalTime]);
    targ{cond}(:,1:end-1) = nan;
end
fitnessFunInputs = targ;
end