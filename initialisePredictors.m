function pred = initialisePredictors(nPred, inputsSet, env)
% inputsSetDim  = numel(inputsSet);
% inputSize     = randi(inputsSetDim);
% inputMask     = randi(inputsSetDim,1,inputSize);
% inputMask     = sort(unique(inputMask));
% inputSize     = numel(inputMask);
% 
% outputSize    = randi(env.dimO);
% outputMask    = randi(env.dimO,1,outputSize);
% outputMask    = sort(unique(outputMask));
% outputSize    = numel(outputMask);
% 

pred(1) = FFN([1 9], [1], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1

%randomly generated
probInput = 0.4*ones(size(inputsSet));

for iPred=1:env.dimO
    out =  mod(iPred-1,env.dimO)+1;
    r = rand();
    if r<0.25
        % 1 layer small
        pred(iPred) = FFN(inputsSet, out, 5, inputsSet,1); % testing the influence of the hidden layer size
    elseif r<0.5
        % 1 layer big
        pred(iPred) = FFN(inputsSet, out, 50, inputsSet,1); % testing the influence of the hidden layer size
    elseif r<0.75
        % 2 layers small
        pred(iPred) = FFN(inputsSet, out, [5 5], inputsSet,1); % testing the influence of the hidden layer size
    else
        % 2 layers big
        pred(iPred) = FFN(inputsSet, out, [50 50], inputsSet,1); % testing the influence of the hidden layer size
    end
end

for iPred=env.dimO+1:2*env.dimO
    out =  mod(iPred-1,env.dimO)+1;
    % 1 layer big
    r = rand();
    if r<0.25
        % 1 layer small
        pred(iPred) = FFN(inputsSet, out, 5, inputsSet,1); % testing the influence of the hidden layer size
    elseif r<0.5
        % 1 layer big
        pred(iPred) = FFN(inputsSet, out, 50, inputsSet,1); % testing the influence of the hidden layer size
    elseif r<0.75
        % 2 layers small
        pred(iPred) = FFN(inputsSet, out, [5 5], inputsSet,1); % testing the influence of the hidden layer size
    else
        % 2 layers big
        pred(iPred) = FFN(inputsSet, out, [50 50], inputsSet,1); % testing the influence of the hidden layer size
    end
end



%{
% 2 layers input mask
 pred(1) = FFN([51], [1], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(2) = FFN([1 2], [2], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(3) = FFN([1 ], [3], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(4) = FFN([51], [4], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(5) = FFN([51 52], [5], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(6) = FFN([51], [6], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(7) = FFN([1 2], [7], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(8) = FFN([1 ], [8], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(9) = FFN([51 52], [9], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(10) = FFN([51 52], [10], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(11) = FFN([51], [11], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(12) = FFN([1 2], [12], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(13) = FFN([1 ], [13], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(14) = FFN([51], [14], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(15) = FFN([51 52], [15], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(16) = FFN([6], [16], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(17) = FFN([1 2], [17], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(18) = FFN([1 ], [18], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(19) = FFN([6], [19], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(20) = FFN([6 7], [20], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(11) = FFN([51], [11], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(12) = FFN([1 2], [12], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(13) = FFN([1 ], [13], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(14) = FFN([51], [14], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(15) = FFN([51 52], [5], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(26) = FFN([6], [26], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(27) = FFN([1 2], [27], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(28) = FFN([1 ], [28], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(29) = FFN([6], [29], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(30) = FFN([6 7], [30], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(31) = FFN([6], [31], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(32) = FFN([1 2], [32], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(33) = FFN([1 ], [33], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(34) = FFN([6], [34], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(35) = FFN([6 7], [35], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(36) = FFN([6], [36], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(37) = FFN([1 2], [37], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(38) = FFN([1 ], [38], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(39) = FFN([6], [39], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(40) = FFN([6 7], [40], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(41) = FFN([6], [41], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(42) = FFN([1 2], [42], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(43) = FFN([1 ], [43], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(44) = FFN([6], [44], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(45) = FFN([6 7], [45], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(46) = FFN([6], [46], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(47) = FFN([1 2], [47], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(48) = FFN([1 ], [48], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(49) = FFN([6], [49], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(50) = FFN([6 7], [50], [5 5], inputsSet,1);  % good for env4 [s1 m1] -> s1

 %1 layer input mask
 pred(101) = FFN([51], [1], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(102) = FFN([1 2], [2], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(103) = FFN([1 ], [3], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(104) = FFN([51], [4], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(105) = FFN([51 52], [5], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(106) = FFN([51], [6], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(107) = FFN([1 2], [7], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(108) = FFN([1 ], [8], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(109) = FFN([51 52], [9], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(110) = FFN([51 52], [10], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(111) = FFN([51], [11], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(112) = FFN([1 2], [12], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(113) = FFN([1 ], [13], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(114) = FFN([51], [14], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(115) = FFN([51 52], [15], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(116) = FFN([6], [16], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(117) = FFN([1 2], [17], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(118) = FFN([1 ], [18], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(119) = FFN([6], [19], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(120) = FFN([6 7], [20], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(111) = FFN([51], [11], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(112) = FFN([1 2], [12], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(113) = FFN([1 ], [13], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(114) = FFN([51], [14], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(115) = FFN([51 52], [5], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(126) = FFN([6], [26], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(127) = FFN([1 2], [27], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(128) = FFN([1 ], [28], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(129) = FFN([6], [29], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(130) = FFN([6 7], [30], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(131) = FFN([6], [31], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(132) = FFN([1 2], [32], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(133) = FFN([1 ], [33], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(134) = FFN([6], [34], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(135) = FFN([6 7], [35], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(136) = FFN([6], [36], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(137) = FFN([1 2], [37], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(138) = FFN([1 ], [38], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(139) = FFN([6], [39], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(140) = FFN([6 7], [40], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(141) = FFN([6], [41], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(142) = FFN([1 2], [42], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(143) = FFN([1 ], [43], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(144) = FFN([6], [44], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(145) = FFN([6 7], [45], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(146) = FFN([6], [46], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(147) = FFN([1 2], [47], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(148) = FFN([1 ], [48], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(149) = FFN([6], [49], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(150) = FFN([6 7], [50], 10, inputsSet,1);  % good for env4 [s1 m1] -> s1
%}

%{ 
%TEST hidden layer size (1 hidden layer)
 pred(1) = FFN([1 9], [1], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1

%randomly generated
probInput = 0.4*ones(size(inputsSet));

for iPred=1:nPred
    out =  mod(iPred-1,env.dimO)+1;
    pred(iPred) = FFN(inputsSet, out, 10^(floor(iPred/env.dimO)+1), inputsSet,1); % testing the influence of the hidden layer size
end
%}

end