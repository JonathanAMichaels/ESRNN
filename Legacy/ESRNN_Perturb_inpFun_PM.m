function [inp, fitnessFunInputs, targetFunPassthrough] = ESRNN_Perturb_inpFun_PM

numConds = 16;
moveTime = 100;
preTime = 80;
totalTime = preTime + moveTime;

allEndpoints{1,1} = [0; 0.5];
allEndpoints{1,2} = [0; 1];
allEndpoints{2,1} = [0.5; 0];
allEndpoints{2,2} = [1; 0];
allEndpoints{3,1} = [0; -0.5];
allEndpoints{3,2} = [0; -1];
allEndpoints{4,1} = [-0.5; 0];
allEndpoints{4,2} = [-1; 0];

allPerturbDir = [0 pi/2 pi pi/2*3];

startPoint = [0; 0];

%% General inputs and output
inp = cell(1,numConds);
targ = cell(1,numConds);
condCount = 1;
for reachDir = 1:4
    for reachDist = 1:2
        endPoint = allEndpoints{reachDir,reachDist};
        for perturbDir = 1:4
            thisPerturbDir = allPerturbDir(perturbDir);
            for perturbMag = 1:3
                
                targetFunPassthrough(condCount).perturbMag = 0.1 * (perturbMag-1);
                
                inp{condCount} = zeros(2, totalTime);
                inp{condCount}(1:2,:) = repmat(endPoint, [1 totalTime]);
                %  inp{condCount}(3,:) = [ones(1,preTime-10) zeros(1,moveTime+10)];
                targ{condCount} = [repmat(startPoint, [1 preTime]), repmat(endPoint, [1 moveTime])];
                targ{condCount}(:,2:end-1) = nan;
                
                targetFunPassthrough(condCount).perturbDir = thisPerturbDir;
                targetFunPassthrough(condCount).perturbTrials = 1;
                
                targetFunPassthrough(condCount).perturbDist = 0.1;
                targetFunPassthrough(condCount).kinStart = preTime;
                targetFunPassthrough(condCount).pos = startPoint;
                targetFunPassthrough(condCount).end = endPoint;
                
                condCount = condCount + 1;
            end
        end
    end
end

fitnessFunInputs = targ;

end