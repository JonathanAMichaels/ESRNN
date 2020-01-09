function [z, targetFeedforward] = ESRNN_CenterOutPerturb_TargetFun(t, r, targetFunPassthrough, targetFeedforward)
cond = targetFunPassthrough.cond;
sPos = targetFunPassthrough.pos(:,cond);
kinStart = targetFunPassthrough.kinStart;
perturbDir = targetFunPassthrough.perturbDir{cond};
targ = targetFunPassthrough.end(:,cond);
if t >= kinStart
    if isempty(targetFeedforward)
        pos = sPos';
        targetFeedforward.t = [];
        targetFeedforward.pos = [];
        targetFeedforward.pON = false;
        targetFeedforward.Feedback = [];
        targetFeedforward.FeedbackHistory = [];
        F = [0, 0];
    else
        pos = targetFeedforward.pos(end,:);
        d = sqrt(sum((pos - targ').^2));
        if (sum(isnan(perturbDir)) == 0) && (d < 0.9)
            targetFeedforward.pON = true;
        end
        if targetFeedforward.pON
            F = [sin(perturbDir), cos(perturbDir)]/50;
        else
            F = [0, 0];
        end
        pos = pos + r'/10 + F;
    end
    targetFeedforward.t(end+1) = t;
    targetFeedforward.pos(end+1,:) = pos;
    z = pos';
    targetFeedforward.Feedback = [z; F'*100];
    targetFeedforward.FeedbackHistory(end+1,:) = targetFeedforward.Feedback;
else
    z = sPos;
end
end