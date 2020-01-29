function net = ESRNN_Context_create_model(policyInitInputs, varargin)

try
    if ispc
        load('E:\jmichaels\Projects\P\Perturb_Best.mat','net')
    else
        load('/Users/jonathanamichaels/Desktop/jmichaels/Projects/P/Perturb_Best.mat', 'net');
    end
catch
    load('/home/jmichae/Perturb_Best.mat', 'net');
end

[wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, net.theta);

wIn = cat(2, wIn, randn(size(wIn,1),1)/sqrt(net.N));

net.I = 4;
net.theta = ESRNN_pack(wIn,J,wOut,x0,bJ,bOut,wFb);

end