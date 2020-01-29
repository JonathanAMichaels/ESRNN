function net = ESRNN_CenterOutPerturb_create_model(policyInitInputs, varargin)

% net = geneticRNN_create_model(policyInitInputs, varargin)
%
% This function initializes a recurrent neural network for later training
% and execution
%
% INPUTS:
%
% N -- the number of recurrent neurons in network
%
% B -- the number of outputs
%
% I -- the number of inputs
%
% p -- the sparseness of the J (connectivity) matrix, (range: 0-1)
%
% g -- the spectral scaling of J
%
% dt -- the integration time constant
%
% tau -- the time constant of each neuron
%
%
% OPTIONAL INPUTS:
%
% actFun -- the activation function used to tranform activations into
% firing rates
% Default: 'tanh'
%
% netNoiseSigma - the variance of random gaussian noise added at each time
% point
% Default: 0
%
% feedback -- whether or not to feed the output of the plant back
% Default: false
%
%
% OUTPUTS:
%
% net -- the network structure
%
%
% Copyright (c) Jonathan A Michaels 2018
% Stanford University
% jonathanamichaels AT gmail DOT com


N = policyInitInputs{1};
B = policyInitInputs{2};
I = policyInitInputs{3};
F = policyInitInputs{4};
p = policyInitInputs{5};
g = policyInitInputs{6};
dt = policyInitInputs{7};
tau = policyInitInputs{8};

if ~isempty(varargin)
    varargin = varargin{1};
end

actFunType = 'tanh'; % Default activation function
netNoiseSigma = 0.0; % Default noise-level
optargin = size(varargin,2);

for i = 1:2:optargin
    switch varargin{i}
        case 'actFun'
            actFunType = varargin{i+1};
        case 'netNoiseSigma'
            netNoiseSigma = varargin{i+1};      
    end
end

%% Assertions
assert(p >= 0 && p <= 1, 'Sparsity must be between 0 and 1.')

%% Initialize internal connectivity
% Connectivity is normally distributed, scaled by the size of the network,
% the sparity, and spectral scaling factor, g.
J = zeros(N,N);

areaInd{1} = 1 : round(N*0.50);
areaInd{2} = round(N*0.50)+1 : N;

interArea = 0.2;
p = [1 interArea; interArea 1];

for area1 = 1:length(areaInd)
    for area2 = 1:length(areaInd)
        thisN = length(areaInd{area1});
        for i = 1:length(areaInd{area1})
            for j = 1:length(areaInd{area2})
                if rand <= p(area1,area2)
                    J(areaInd{area1}(i),areaInd{area2}(j)) = g * randn / sqrt(p(area1,area2) * thisN);
                end
            end
        end
    end
end

e = abs(eig(J));
J = J / e(1) * g;

net.I = I;
net.B = B;
net.N = N;
net.F = F;
net.p = p;
net.g = g;
net.netNoiseSigma = netNoiseSigma;
net.dt = dt;
net.tau = tau;

%% Initialize input weights
wIn = zeros(N,I);
wIn(areaInd{1},:) = randn(length(areaInd{1}),I) / sqrt(length(areaInd{1}));
%wIn = wIn * 1e-1;

%% Initialize feedback weights
wFb = zeros(N,F);
if F > 0
    wFb(areaInd{2},:) = randn(length(areaInd{2}),F) / sqrt(length(areaInd{2}));
  %  wFb = wFb * 1e-1;
end

%% Initialize output weights
wOut = zeros(B,N);
wOut(1:2,areaInd{1}) = randn(2,length(areaInd{1})) / sqrt(length(areaInd{1}));
wOut(3,areaInd{2}) = randn(1,length(areaInd{2})) / sqrt(length(areaInd{2}));
%wOut = wOut * 1e-2;

%% Initialize J biases
bJ = randn(N,1) * 1e-6;

%% Initialize output biases
bOut = randn(B,1) * 1e-6;

%% Initialize starting activation
x0 = randn(N,1) * 1e-6;

%% Pack it up
net.theta = ESRNN_pack(wIn, J, wOut, x0, bJ, bOut, wFb);

%% Activation function
switch actFunType
    case 'tanh'
        net.actFun = @tanh;
    case 'recttanh'
        net.actFun = @(x) (x > 0) .* tanh(x);
    case 'baselinetanh' % Similar to Rajan et al. (2010)
        net.actFun = @(x) (x > 0) .* (1 - 0.1) .* tanh(x / (1 - 0.1)) ...
            + (x <= 0) .* 0.1 .* tanh(x / 0.1);
    case 'linear'
        net.actFun = @(x) x;
    otherwise
        assert(false, 'Nope!');
end
end