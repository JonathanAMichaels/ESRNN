   function fitness = ESRNN_COFitnessFun(Z0, Z1, kin, targ)
        fitness = zeros(1,length(Z1));
        for cond = 1:length(Z1)
            ind = ~isnan(targ{cond});
            useZ1 = Z1{cond}(ind);
            useF = targ{cond}(ind);
            
            d = sqrt(mean((useZ1(:)-useF(:)).^2));
            if d < 0.09
                err(1) = 0;
            else
                err(1) = d;
            end
            err(2) = sqrt(mean((Z0{cond}(:).^2)))*2;
            fitness(cond) = -sum(err);
        end
    end