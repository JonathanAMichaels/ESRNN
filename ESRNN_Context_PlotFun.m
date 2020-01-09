function ESRNN_Perturb_PlotFun(plotStats, errStats, evalOptions)
if evalOptions(1) >= 0
    disp(['Generation: ' num2str(plotStats.generation) '  Fitness: ' num2str(mean(plotStats.fitness(:,1))) '  Time Required: ' num2str(plotStats.tRun) ' seconds'])
end
if evalOptions(1) >= 1
    figure(98)
    set(gcf, 'Name', 'Error', 'NumberTitle', 'off')
    c = lines(size(plotStats.fitness,1));
    for type = 1:size(plotStats.fitness,1)
        h1(type) = plot(plotStats.generation, plotStats.fitness(type,1), '.', 'MarkerSize', 20, 'Color', c(type,:));
        hold on
    end
    plot(plotStats.generation, mean(plotStats.fitness(:,1),1), '.', 'MarkerSize', 40, 'Color', [0 0 0]);
    set(gca, 'XLim', [1 plotStats.generation+0.1])
    xlabel('Generation')
    ylabel('Fitness')
end
if evalOptions(1) >= 2
    figure(99)
    set(gcf, 'Name', 'Output', 'NumberTitle', 'off')
    clf
    subplot(3,1,1)
    hold on
%    h1 = filledCircle([plotStats.targ{condCount}(1,1) plotStats.targ{condCount}(2,1)], 0.1, 100, [1 1 1]);

    rectangle('Pos', [-0.05 0.9750 0.1 0.1])
    rectangle('Pos', [-0.8 0.9750 0.2 0.1])
    rectangle('Pos', [0.6 0.9750 0.2 0.1])
    
    c = lines(length(plotStats.bigZ1));
    for condCount = 1:length(plotStats.bigZ1)
     %   h2 = filledCircle([plotStats.targ{condCount}(1,end) plotStats.targ{condCount}(2,end)], 0.1, 100, [0.9 0.9 0.9]);
     %   h1.EdgeColor = c(condCount,:);
     %   h2.EdgeColor = c(condCount,:);
    end
    for condCount = 1:length(plotStats.bigZ1)
        plot(plotStats.bigZ1{condCount}(1,:), plotStats.bigZ1{condCount}(2,:), 'Color', c(condCount,:));
        
      %  plot(plotStats.bigZ1{condCount}(1, plotStats.kin(condCount).goTime), ...
    %    plotStats.bigZ1{condCount}(2, plotStats.kin(condCount).goTime), '.', 'Color', c(condCount,:), 'MarkerSize', 20)
    end
    axis([-1.8 1.8 -0.5 1.8])
    axis square
    subplot(3,1,2)
    hold on
    for condCount = 1:length(plotStats.bigZ1)
        plot(plotStats.bigZ1{condCount}', 'Color', c(condCount,:));
    end
    subplot(3,1,3)
    hold on
    for condCount = 1:length(plotStats.bigZ1)
        plot(plotStats.bigR{condCount}(1,:)', 'Color', c(condCount,:));
    end
end
drawnow
end