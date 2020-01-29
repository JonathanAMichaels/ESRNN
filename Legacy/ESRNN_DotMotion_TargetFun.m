function [z, targetFeedforward] = ESRNN_Perturb_TargetFun(t, r, targetFunPassthrough, targetFeedforward)

dt = targetFunPassthrough.dt;
startPoint = targetFunPassthrough.pos';
kinStart = targetFunPassthrough.kinStart;
endPoint = targetFunPassthrough.end';
perturbTrials = targetFunPassthrough.perturbTrials;
perturbDir = targetFunPassthrough.perturbDir;
perturbMag = targetFunPassthrough.perturbMag;
perturbDist = targetFunPassthrough.perturbDist;
if t == 0
    pos1 = startPoint;
    vel1 = [0 0];
    F1 = [0 0];
    FOut1 = [0 0];
    
    targetFeedforward.t = [];
    targetFeedforward.pos = [];
    targetFeedforward.vel = [];
    targetFeedforward.F = [];
    targetFeedforward.FOut = [];
    targetFeedforward.pON = false;
    targetFeedforward.Feedback = [];
    targetFeedforward.FeedbackHistory = [];
    targetFeedforward.perturbDir = [];
    targetFeedforward.perturbOnTime = [];
    targetFeedforward.perturbMag = [];
    targetFeedforward.perturbDist = [];
    targetFeedforward.inTarg = [];
    targetFeedforward.kinStart = kinStart;
    targetFeedforward.lock = false;
    targetFeedforward.lockTime = [];
    targetFeedforward.inpOff = [];
else
    vel1 = targetFeedforward.vel(end,:);
    F1 = targetFeedforward.F(end,:);
    FOut1 = targetFeedforward.FOut(end,:);
    pos1 = targetFeedforward.pos(end,:);
end

FOut = tanh(r');

if t >= kinStart && ~targetFeedforward.lock
    %% Calculate forces
    dToStart = sqrt(sum((startPoint - pos1).^2));
    dToEnd = sqrt(sum((endPoint - pos1).^2));
    dTrigger = sqrt(sum((endPoint - startPoint).^2)) * (1 - perturbDist);
    if dToStart > 0.1 || ~isempty(targetFeedforward.inpOff)
        targetFeedforward.inpOff = 1;
    end
    dToLeft = sqrt(sum(([-1 0] - pos1).^2));
    dToRight = sqrt(sum(([1 0] - pos1).^2));
    if dToLeft < 0.1 || dToRight < 0.1
        targetFeedforward.inTarg(end+1) = t;
        if length(targetFeedforward.inTarg) >= 20
            conseq = diff(targetFeedforward.inTarg(end-19:end));
            if sum(conseq == 1) == 19
          %      targetFeedforward.lock = true;
          %      targetFeedforward.lockTime = t;
            end
        end         
    end

    F = [0 0];
    if targetFeedforward.pON
        F = [sin(targetFeedforward.perturbDir), cos(targetFeedforward.perturbDir)] * perturbMag;
    end
  
    if ~targetFeedforward.lock
        %% Update current velocity
        outputDelay = 3;
        if size(targetFeedforward.FOut,1) < outputDelay
            oInd = size(targetFeedforward.FOut,1);
        else
            oInd = outputDelay;
        end
        if isempty(targetFeedforward.FOut)
            thisFOut = FOut;
        else
            thisFOut = targetFeedforward.FOut(end-(oInd-1),:);
        end
        vel = vel1 + (thisFOut + F) * (dt/1000);
        pos = pos1 + vel;
    else
        pos = pos1;
        vel = 0;
        FOut = 0;
        F = 0;
    end
else
    pos = pos1;
    vel = vel1;
    F = F1;
end
targetFeedforward.vel(end+1,:) = vel;
targetFeedforward.F(end+1,:) = F;
targetFeedforward.FOut(end+1,:) = FOut;
targetFeedforward.t(end+1) = t;
targetFeedforward.pos(end+1,:) = pos;
feedbackDelay = 3;
if size(targetFeedforward.pos,1) < feedbackDelay+1
    FInd = size(targetFeedforward.pos,1) - 1;
else
    FInd = feedbackDelay;
end
targetFeedforward.Feedback = [targetFeedforward.pos(end-FInd,:)'; targetFeedforward.FOut(end-FInd,:)'];
targetFeedforward.Feedback = targetFeedforward.Feedback;
targetFeedforward.FeedbackHistory(end+1,:) = targetFeedforward.Feedback;
z = pos';
end