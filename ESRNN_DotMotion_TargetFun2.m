    function [z, targetFeedforward] = defaultTargetFunction(t, r, ~, targetFeedforward)
        z = r; % Just passes firing rate
        
        if t == 0
            targetFeedforward.inpOff = [];
        end
        
        d = sqrt(sum((z - [0;0]).^2));
        if d > 0.1 || ~isempty(targetFeedforward.inpOff)
        %    targetFeedforward.inpOff = 1;
        end
            
        
        
        targetFeedforward.Feedback = z;
    end