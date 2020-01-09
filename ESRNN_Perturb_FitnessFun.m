function fitness = ESRNN_Perturb_FitnessFun(Z0, Z1, kin, targ)
fitness = zeros(1,length(Z1));
for cond = 1:length(Z1)
    ind = ~isnan(targ{cond}(1,:));
    useZ1 = Z1{cond}(1:2,ind);
    useF = targ{cond}(1:2,ind);

    d = zeros(1,size(useZ1,2));
    for t = 1:size(useZ1,2)
        d(t) = sqrt(sum((useZ1(:,t) - useF(:,t)).^2));
    end
    d(d < 0.1) = 0;
    
    err(1) = mean(d);
    tRange = 1:size(Z0{cond},2);
    err(2) = sqrt(sum(sum(Z0{cond}(1:2,tRange).^2))) / size(Z0{cond},2) * 2;
  
    fitness(cond) = -sum(err);
end
end


    