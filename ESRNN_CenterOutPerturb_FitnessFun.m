function fitness = ESRNN_CenterOutPerturb_FitnessFun(Z0, Z1, kin, targ)
fitness = zeros(1,length(Z1));
for cond = 1:length(Z1)
    ind = ~isnan(targ{cond});
    useZ0 = Z0{cond}(:,21:end);
    useZ1 = Z1{cond}(ind);
    useF = targ{cond}(ind);
    
    err(1) = sqrt(mean((useZ1(:)-useF(:)).^2)) - 0.07;
    if err(1) < 0
        err(1) = 0;
    end
    err(2) = sqrt(mean(useZ0(:).^2)) * 0.5 * 0;
    err(3) = sqrt(mean(kin(cond).vel(end).^2)) * 0;
    fitness(cond) = -sum(err);
end

end
