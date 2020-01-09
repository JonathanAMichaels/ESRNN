function [z, targetFeedforward] = ESRNN_CenterOutPerturb_Arm_TargetFun(t, r, targetFunPassthrough, targetFeedforward)

L = targetFunPassthrough.L;
cond = targetFunPassthrough.cond;
dt = targetFunPassthrough.dt;
%startPos = targetFunPassthrough.pos(:,cond);
kinStart = targetFunPassthrough.kinStart;
perturbDir = targetFunPassthrough.perturbDir{cond};
targ = targetFunPassthrough.end(:,cond)';

if t == 0
    %% Set initial angles
    ang1 = [pi/10*8 (2*(pi - pi/10*8))];
    initvals(1) = -(L(1)*cos(ang1(1)+pi) + L(2)*cos(ang1(2)+ang1(1)));
    initvals(2) = -(L(1)*sin(ang1(1)) + L(2)*sin(ang1(2)+ang1(1)-pi));
    
    %% Calculate endpoint
    pos(1) = initvals(1) + L(1)*cos(ang1(1)+pi) + L(2)*cos(ang1(2)+ang1(1));
    pos(2) = initvals(2) + L(1)*sin(ang1(1)) + L(2)*sin(ang1(2)+ang1(1)-pi);
    
    %% Calculate eblow position for saving
    posL1(1) = initvals(1) + L(1)*cos(ang1(1)+pi);
    posL1(2) = initvals(2) + L(1)*sin(ang1(1));
    
    vel1 = [0 0];
    F1 = [0 0];
    
    targetFeedforward.initvals = initvals;
    targetFeedforward.t = [];
    targetFeedforward.posL1 = [];
    targetFeedforward.pos = [];
    targetFeedforward.ang = [];
    targetFeedforward.vel = [];
    targetFeedforward.F = [];
    targetFeedforward.pON = false;
    targetFeedforward.Feedback = [];
    targetFeedforward.FeedbackHistory = [];
else
    ang1 = targetFeedforward.ang(end,:);
    vel1 = targetFeedforward.vel(end,:);
    F1 = targetFeedforward.F(end,:);
    initvals = targetFeedforward.initvals;
    posL1 = targetFeedforward.posL1(end,:);
    pos = targetFeedforward.pos(end,:);
end

if t >= kinStart
    %% Update current velocity
    vel = zeros(1,length(r));
    for d = 1:length(r)
        vel(d) = vel1(d) + r(d)*(dt/1000) + F1(d);
    end
    ang = zeros(1,length(r));
    for d = 1:length(r)
        ang(d) = ang1(d) + vel(d);
    end
    
    %% Set joint limits
    if ang(1) > pi/2*3
        ang(1) = pi/2*3;
        vel(1) = 0;
    elseif ang(1) < 0
        ang(1) = 0;
        vel(1) = 0;
    end
    if ang(2) > (pi - pi/20)
        ang(2) = (pi - pi/20);
        vel(2) = 0;
    elseif ang(2) < pi/20
        ang(2) = pi/20;
        vel(2) = 0;
    end
    
    %% Calculate endpoint
    pos(1) = initvals(1) + L(1)*cos(ang(1)+pi) + L(2)*cos(ang(2)+ang(1));
    pos(2) = initvals(2) + L(1)*sin(ang(1)) + L(2)*sin(ang(2)+ang(1)-pi);
    
    %% Calculate eblow position for saving
    posL1(1) = initvals(1) + L(1)*cos(ang(1)+pi);
    posL1(2) = initvals(2) + L(1)*sin(ang(1));
    
    %% Calculate forces
    d = sqrt(sum((pos - targ).^2));
    if (sum(isnan(perturbDir)) == 0) && (d < 0.9)
        targetFeedforward.pON = true;
    end
    if targetFeedforward.pON
        F = [sin(perturbDir), cos(perturbDir)] * (dt/1000) / 2; % best was 2!!
        initAng = ESRNN_CenterOutPerturb_InverseKin(pos, L, initvals);
        pAng = ESRNN_CenterOutPerturb_InverseKin(pos+F, L, initvals);
        F = pAng - initAng;
        %    F = [0 0];
    else
        F = [0 0];
    end  
else
    ang = ang1;
    vel = vel1;
    F = F1;
end
targetFeedforward.ang(end+1,:) = ang;
targetFeedforward.vel(end+1,:) = vel;
targetFeedforward.F(end+1,:) = F;
targetFeedforward.t(end+1) = t;
targetFeedforward.posL1(end+1,:) = posL1;
targetFeedforward.pos(end+1,:) = pos;
targetFeedforward.Feedback = [ang1'; F1' / (dt/1000) * 100]; %[ang1'; F1' / (dt/1000) * 100];
targetFeedforward.FeedbackHistory(end+1,:) = targetFeedforward.Feedback;
z = pos';
end