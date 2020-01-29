function ang = ESRNN_CenterOutPerturb_InverseKin(pos, L, shoulder)
elbow = -1;

x = pos(1) - shoulder(1);
y = pos(2) - shoulder(2);

l1 = L(1);
l2 = L(2);

numer = (x.^2 + y.^2 - l1^2 - l2^2) / (2 * l1 * l2);
theta2 = atan2(elbow*sqrt(1 - numer.^2), numer);

k1 = l1 + l2 * cos(theta2);
k2 = l2 * sin(theta2);
theta1 = atan2(y, x) - atan2(k2, k1);

ang = [theta1; theta2];
end

