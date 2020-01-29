    function [z, targetFeedforward] = defaultTargetFunction(~, r, ~, targetFeedforward)
    
            if r >= 0
                z = 1;
            else
                z = -1;
            end
        targetFeedforward.Feedback = z;
    end