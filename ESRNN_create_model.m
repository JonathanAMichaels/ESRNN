function net = ESRNN_create_model(policyInitInputs, varargin)

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
for i = 1:N
    for j = 1:N
        if rand <= p
            J(i,j) = g * randn / sqrt(p*N);
        end
    end
end
e = abs(eig(J));
J = J / e(1) * g;

net.I = I;
net.B = B;
net.N = N;
net.p = p;
net.g = g;
net.F = F;
net.netNoiseSigma = netNoiseSigma;
net.dt = dt;
net.tau = tau;

%% Initialize input weights
wIn = randn(N,I) / sqrt(I);

%% Initialize feedback weights
wFb = zeros(N,F);
if F > 0
    wFb = randn(N,F) / sqrt(F);
end

%% Initialize output weights
wOut = randn(B,N) / sqrt(B) * 1e-50;

%% Initialize J biases
bJ = (rand(N,1)-0.5)*2;

%% Initialize output biases
bOut = zeros(B,1) + 0.1;

%% Initialize starting activation
x0 = (rand(N,1) - 0.5)*2;

%% Pack it up
net.theta = ESRNN_pack(wIn, J, wOut, x0, bJ, bOut, wFb);
net.modMask = ones(size(net.theta));

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