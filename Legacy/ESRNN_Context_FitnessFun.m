function fitness = ESRNN_Perturb_FitnessFun(Z0, Z1, kin, targ)
fitness = zeros(1,length(Z1));
for cond = 1:length(Z1)
    useZ1 = Z1{cond}(:,end);
    useF = targ{cond}(:,end);
    
    thisCond = kin(cond).thisCond;
    if thisCond == 1 || ((thisCond == 2) && ~kin(cond).pON)
        if useZ1(1) < -0.05
            d = sqrt(sum((useZ1 - [-0.05; 1]).^2));
        elseif useZ1(1) > 0.05
            d = sqrt(sum((useZ1 - [0.05; 1]).^2));
        else
            d = sqrt(sum((useZ1(2) - 1).^2));
        end
    elseif kin(cond).perturbDir == 3*pi/2
        if useZ1(1) < -0.8
            d = sqrt(sum((useZ1 - [-0.8; 1]).^2));
        elseif useZ1(1) > -0.6
            d = sqrt(sum((useZ1 - [-0.6; 1]).^2));
        else
            d = sqrt(sum((useZ1(2) - 1).^2));
        end
    elseif kin(cond).perturbDir == pi/2
        if useZ1(1) < 0.6
            d = sqrt(sum((useZ1 - [0.6; 1]).^2));
        elseif useZ1(1) > 0.8
            d = sqrt(sum((useZ1 - [0.8; 1]).^2));
        else
            d = sqrt(sum((useZ1(2) - 1).^2));
        end
    end
    
    err(1) = sum(d);
    err(2) = sqrt(sum(Z0{cond}(:).^2)) / size(Z0{cond},2);
    err(3) = sum(Z1{cond}(2,:) < 0) / size(Z1{cond},2);

    fitness(cond) = -sum(err);
end
end
