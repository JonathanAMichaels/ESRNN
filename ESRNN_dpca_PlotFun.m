function dpca_plot_default(data, time, yspan, explVar, compNum, events, signif, marg)

% Modify this function to adjust how components are plotted.
%
% Parameters are as follows:
%   data      - data matrix, size(data,1)=1 because it's only one component
%   time      - time axis
%   yspan     - y-axis spab
%   explVar   - variance of this component
%   compNum   - component number
%   events    - time events to be marked on the time axis
%   signif    - marks time-point where component is significant
%   marg      - marginalization number


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% displaying legend
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(data, 'legend')
    
    % if there is only time and no other parameter - do nothing
    if length(time) == 2
        return
        
        % if there is one parameter
    elseif length(time) == 3
        numOfStimuli = time(2); % time is used to pass size(data) for legend
        colors = lines(numOfStimuli);
        hold on
        
        for f = 1:numOfStimuli
            plot([0.5 1], [f f], 'color', colors(f,:), 'LineWidth', 2)
            text(1.2, f, ['Stimulus ' num2str(f)])
        end
        axis([0 3 -1 1.5+numOfStimuli])
        set(gca, 'XTick', [])
        set(gca, 'YTick', [])
        set(gca,'Visible','off')
        return
        
        % two parameters: stimulus and decision (decision can only have two
        % values)
    elseif length(time) == 4 && time(3) == 2
        numOfStimuli = time(2); % time is used to pass size(data) for legend
        colors = lines(numOfStimuli);
        hold on
        
        for f = 1:numOfStimuli
            plot([0.5 1], [f f], 'color', colors(f,:), 'LineWidth', 2)
            text(1.2, f, ['Stimulus ' num2str(f)])
        end
        plot([0.5 1], [-2 -2], 'k', 'LineWidth', 2)
        plot([0.5 1], [-3 -3], 'k--', 'LineWidth', 2)
        text(1.2, -2, 'Decision 1')
        text(1.2, -3, 'Decision 2')
        
        axis([0 3 -4.5 1.5+numOfStimuli])
        set(gca, 'XTick', [])
        set(gca, 'YTick', [])
        set(gca,'Visible','off')
        return
        
        % other cases - do nothing
    else
        return
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% setting up the subplot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isempty(time)
    time = 1:size(data, ndims(data));
end
axis([time(1) time(end) yspan])
hold on

if ~isempty(explVar)
    title(['Component #' num2str(compNum) ' [' num2str(explVar,'%.1f') '%]'])
else
    title(['Component #' num2str(compNum)])
end

if ~isempty(events)
    plot([events; events], yspan, 'Color', [0.6 0.6 0.6])
end

if ~isempty(signif)
    signif(signif==0) = nan;
    plot(time, signif + yspan(1) + (yspan(2)-yspan(1))*0.05, 'k', 'LineWidth', 3)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotting the component
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% in all other cases pool all conditions and plot them in different
% colours

%% Plot center-out reaching results
%c = lines(100);
cc = [lines(5); 0 0 0];
c = [];
for i = 1:5+1
    c(end+1:end+5,:) = repmat(cc(i,:), [5 1]);
end
cmap = c;

data = squeeze(data);
data = permute(data, [3 1 2 ]);
data2 = data(:,:);

for cond = 1:size(data2,2)
  %  if ismember(cond, [6:10 16:20])
 %       lineStyle = ':';
  %  else
        lineStyle = '-';
  %  end
    h(cond) = plot(time, data2(:,cond), lineStyle, 'LineWidth', 2, 'Color', cmap(cond,:));
end
if compNum == 20
 %   legend(h, {'Contra Precision -50', 'Contra Precision -25', 'Contra Precision 0', 'Contra Precision 25', 'Contra Precision 50', ...
  %      'Contra Power -50', 'Contra Power -25', 'Contra Power 0', 'Contra Power 25', 'Contra Power 50', ...
   %     'Ipsi Precision -50', 'Ipsi Precision -25', 'Ipsi Precision 0', 'Ipsi Precision 25', 'Ipsi Precision 50', ...
    %    'Ipsi Power -50', 'Ipsi Power -25', 'Ipsi Power 0', 'Ipsi Power 25', 'Ipsi Power 50'}, 'Location', 'SouthEastOutside')
end
%plot([200.1 200.1], get(gca, 'YLim'), 'White', 'LineWidth', 4)
%plot([260.1 260.1], get(gca, 'YLim'), 'Color', 'White', 'LineWidth', 4)

set(gca, 'XTick', [])


