function fitness = ESRNN_Perturb_FitnessFun(Z0, Z1, kin, targ)
fitness = zeros(1,length(Z1));
for cond = 1:length(Z1)
    ind = ~isnan(targ{cond}(1,:));
    useZ1 = Z1{cond}(1:2,ind);
    useF = targ{cond}(1:2,ind);

    d = zeros(1,size(Z1{cond},2));
    for t = 1:size(Z1{cond},2)
        d(t) = sqrt(sum((Z1{cond}(:,t) - targ{cond}(:,end)).^2));
    end
    d(d < 0.05) = 0;
    d = d(end);
    
    err(1) = d;
    tRange = 1:size(Z0{cond},2);
    err(2) = sqrt(sum(sum(Z0{cond}(1:2,tRange).^2))) / size(Z0{cond},2) * 2;
    err(3) = sum(Z1{cond}(2,:) < 0);
    
    fitness(cond) = -sum(err);
end
end


    