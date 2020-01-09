function theta = ESRNN_pack(wIn, J, wOut, x0, bJ, bOut, wFb)


theta = [wIn(:); J(:); wOut(:); x0(:); bJ(:); bOut(:); wFb(:)];
