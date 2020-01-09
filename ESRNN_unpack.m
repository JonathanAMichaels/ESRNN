function [wIn, J, wOut, x0, bJ, bOut, wFb] = ESRNN_unpack(net, theta)

W = cell(1,7);
ind = 1;
for p = 1:7
    if p == 1
        i = net.N;
        j = net.I;
    elseif p == 2
        i = net.N;
        j = net.N;
    elseif p == 3
        i = net.B;
        j = net.N;
    elseif p == 4 || p == 5
        i = net.N;
        j = 1;
    elseif p == 6
        i = net.B;
        j = 1;
    elseif p == 7
        i = net.N;
        j = net.F;
    end
    L = i*j;
    W{p} = reshape(theta(ind : ind + L - 1), [i j]);
    ind = ind + L;
end

wIn = W{1};
J = W{2};
wOut = W{3};
x0 = W{4};
bJ = W{5};
bOut = W{6};
wFb = W{7};

end