function fitness = ESRNN_Perturb_FitnessFun(Z0, Z1, kin, targ)
fitness = zeros(1,length(Z1));
for cond = 1:length(Z1)
    ind = ~isnan(targ{cond}(1,:));
    useZ1 = Z1{cond}(1:2,ind);
    useF = targ{cond}(1:2,ind);

    d = sqrt(sum((useZ1 - useF).^2));
    d(d < 0.1) = 0;
    
    err(1) = sum(d);
    tRange = 1:size(Z0{cond},2);
    err(2) = sqrt(sum(sum(Z0{cond}(1:2,tRange).^2))) / size(Z0{cond},2);

    if ~isempty(kin(cond).perturbOnTime)
        pOn = kin(cond).perturbOnTime+10;
        if pOn > size(targ{cond},2)
            pOn = size(targ{cond},2);
        end
        thisTarg = [zeros(1,pOn)-1 ones(1,size(targ{cond},2)-pOn)];
    else
        thisTarg = zeros(1,size(targ{cond},2));
    end
    err(3) = sqrt(sum((Z0{cond}(3,:) - thisTarg).^2)) / length(thisTarg) * 10;
  
    fitness(cond) = -sum(err);
end
end


    