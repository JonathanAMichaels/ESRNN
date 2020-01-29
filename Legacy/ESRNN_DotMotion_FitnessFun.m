function fitness = ESRNN_DotMotion_FitnessFun(Z0, Z1, kin, targ)
fitness = zeros(1,length(Z1));
for cond = 1:length(Z1)
     ind = ~isnan(targ{cond}(1,:));
     initInd = find(targ{cond}(1,:) == 0);
  %   ind = randi(size(targ{cond},2));
     
    useZ1 = Z1{cond}(:,ind);
    useF = targ{cond}(:,ind);
    bigZ1 = Z1{cond};
    
    d = zeros(1,size(useF,2));
    for t = 1:size(useF,2)
        d(t) = sqrt(sum((useZ1(:,t)-useF(:,t)).^2));
    end

    d(d < 0.1) = 0;
    d(d(initInd) > 0) = 1;

    err(1) = mean(d);
    err(2) = sqrt(sum(Z0{cond}(:).^2)) / size(Z0{cond},2) * 2;
    
    
    fitness(cond) = -sum(err);
    
 %   choice(cond) = bigZ1(1,end);
end
%extra = -abs(-sum(choice < 0) + sum(choice > 0)) / length(choice);
fitness = fitness;
end