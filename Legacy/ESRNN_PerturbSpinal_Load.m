function net = ESRNN_CenterOutPerturb_create_model(policyInitInputs, varargin)


load('E:\jmichaels\Projects\P\PerturbSpinal_Best.mat', 'net')


[wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, net.theta);
trainedJ = J;

N = net.N;
I = net.I;
g = net.g;

areaInd{1} = 1 : round(N*0.45);
areaInd{2} = round(N*0.45)+1 : round(N*0.90);
areaInd{3} = round(N*0.90)+1 : N;


interArea = 0.04;
p = [1 interArea 0; interArea 1 interArea; interArea 0 1];

for area1 = 1:length(areaInd)
    for area2 = 1:length(areaInd)
        thisN = mean([length(areaInd{area1}) length(areaInd{area2})]);
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

J(areaInd{3}, areaInd{3}) = trainedJ(areaInd{3}, areaInd{3});

wIn = zeros(N,I);
wIn(areaInd{1},:) = randn(length(areaInd{1}),I) / sqrt(length(areaInd{1}));

net.theta = ESRNN_pack(wIn, J, wOut, x0, bJ, bOut, wFb);
modwIn = wIn ~= 0;
modx0 = ones(size(x0));
modx0(areaInd{3}) = 0;
modJ = J ~= 0;
modJ(areaInd{3},areaInd{3}) = 0;
modbJ = ones(size(bJ));
modbJ(areaInd{3}) = 0;
net.modMask = ESRNN_pack(modwIn, modJ, zeros(size(wOut)), modx0, modbJ, zeros(size(bOut)), zeros(size(wFb)));


end