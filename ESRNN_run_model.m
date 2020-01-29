function [Z0, Z1, R, X, varargout] = ESRNN_run_model(net, inp, varargin)

%
% This function runs the network structure (net)
%
%
% INPUTS:
%
% net -- The network structure created by geneticRNN_create_model
%
% inp -- Inputs to the network. Must be present, but can be empty.
%
%
% OUTPUTS:
%
% Z0 -- the output of the network
%
% Z1 -- The output of the plant. This is only different from Z0 if a custom target function is supplied
%
% R -- The firing rate of all neurons in the network
%
% X -- The activation of all neurons in the network
%
% targetOut -- structure containing the output produced by targetFun
% (optional)
%
%
% Copyright (c) Jonathan A Michaels 2018
% Stanford University
% jonathanamichaels AT gmail DOT com


% Variable output considerations
nout = max(nargout,1)-1;

% Variable input considerations
optargin = size(varargin,2);

targetFun = @defaultTargetFunction;
targetFunPassthrough = [];
for iVar = 1:2:optargin
    switch varargin{iVar}
        case 'targetFun'
            targetFun = varargin{iVar+1};
        case 'targetFunPassthrough'
            targetFunPassthrough = varargin{iVar+1};
    end
end

N = net.N;
B = net.B;
I = net.I;
F = net.F;

assert(size(inp{1},1) == I, 'There must be an input entry for each input vector.');
condList = 1:length(inp);

[wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, net.theta);
dt = net.dt;
tau = net.tau;
dt_div_tau = dt/tau;
netNoiseSigma = net.netNoiseSigma;
actFun = net.actFun;

Z1 = cell(1,length(condList));
Z0 = cell(1,length(condList));
R = cell(1,length(condList));
X = cell(1,length(condList));

for cond = 1:length(condList)
    thisCond = condList(cond);
    thisInp = inp{thisCond};
    niters = size(thisInp,2);
    targetFeedforward = [];
    if ~isempty(targetFunPassthrough)
        thisPass = targetFunPassthrough(cond);
        thisPass.dt = dt;
    else
        thisPass = [];
    end
    
    x = x0;
    
    %% Activation function
    r = actFun(x);
    %% Output
    out = wOut*r + bOut;
    
    %% Calculate output using supplied function
    [z, targetFeedforward] = targetFun(0, out, thisPass, targetFeedforward);

    allZ0 = zeros(niters,length(out));
    allZ1 = zeros(niters,length(z));
    allR = zeros(niters,N);
    allX = zeros(niters,N);
    
    for i = 1:niters
        if isfield(targetFeedforward, 'inpOff')
            if ~isempty(targetFeedforward.inpOff)
                thisInp(targetFeedforward.inpOff,:) = 0;
            end
        end
        input = wIn*thisInp(:,i);
        allZ0(i,:) = out;
        allZ1(i,:) = z;
        allR(i,:) = r;
        allX(i,:) = x;
        if i == niters
            saveTarg = targetFeedforward;
        end
        
        if isfield(targetFeedforward, 'Feedback') && F ~= 0
            if ~isempty(targetFeedforward.Feedback)
                Feedback = targetFeedforward.Feedback;
            else
                Feedback = zeros(size(wFb,2),1);
            end
        else
            Feedback = zeros(size(wFb,2),1);
        end

        %% Calculate change in activation
        excitation = -x + J*r + input + wFb*Feedback + bJ + netNoiseSigma*randn(N,1);
        
        %% Add all activation changes together
        x = x + dt_div_tau*excitation;
        
        %% Activation function
        r = actFun(x);
        %% Output
        out = wOut*r + bOut;
        
        %% Calculate output using supplied function
        [z, targetFeedforward] = targetFun(i, out, thisPass, targetFeedforward);
    end
    %% Save all states
    Z0{cond} = allZ0';
    Z1{cond} = allZ1';
    R{cond} = allR';
    X{cond} = allX';
    if ~isempty(saveTarg)
        targetOut(cond) = saveTarg;
    end
end

%% Output error statistics if required
if (nout >= 4)
    if exist('targetOut', 'var')
        varargout{1} = targetOut;
    else
        varargout{1} = [];
    end
end

%% Default output function
    function [z, targetFeedforward] = defaultTargetFunction(~, r, ~, targetFeedforward)
        z = r; % Just passes firing rate
        targetFeedforward.Feedback = z;
    end
end