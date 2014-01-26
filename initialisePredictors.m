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

for iPred=1:nPred
    out =  mod(iPred-1,env.dimO)+1;
    %delay = floor((iPred-1)/env.dimO)+1;
    pred(iPred) = FFN(inputsSet, out, 20, inputsSet,1); % testing the influence of the hidden layer size
    %[pred(iPred), inPredi, outPredi] = generatePredictor(inputsSet, pred, env.dimO,out, delay, probInput);
end
 pred(1) = FFN([51], [1], [20 20], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(2) = FFN([1 2], [2], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(3) = FFN([1 ], [3], [10 10], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(4) = FFN([51], [4], [10 10], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(5) = FFN([51 52], [5], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(6) = FFN([51], [6], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(7) = FFN([1 2], [7], 3, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(8) = FFN([1 ], [8], 100, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(9) = FFN([51], [9], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(10) = FFN([51 52], [10], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(11) = FFN([51], [11], [20 20], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(12) = FFN([1 2], [12], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(13) = FFN([1 ], [13], [10 10], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(14) = FFN([51], [14], [10 10], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(15) = FFN([51 52], [15], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(16) = FFN([6], [16], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(17) = FFN([1 2], [17], 3, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(18) = FFN([1 ], [18], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(19) = FFN([6], [19], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(20) = FFN([6 7], [20], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(11) = FFN([51], [11], [20 20], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(12) = FFN([1 2], [12], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(13) = FFN([1 ], [13], [10 10], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(14) = FFN([51], [14], [10 10], inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(15) = FFN([51 52], [5], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(26) = FFN([6], [26], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(27) = FFN([1 2], [27], 3, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(28) = FFN([1 ], [28], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(29) = FFN([6], [29], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(30) = FFN([6 7], [30], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(31) = FFN([6], [31], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(32) = FFN([1 2], [32], 3, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(33) = FFN([1 ], [33], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(34) = FFN([6], [34], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(35) = FFN([6 7], [35], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(36) = FFN([6], [36], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(37) = FFN([1 2], [37], 3, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(38) = FFN([1 ], [38], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(39) = FFN([6], [39], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(40) = FFN([6 7], [40], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(41) = FFN([6], [41], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(42) = FFN([1 2], [42], 3, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(43) = FFN([1 ], [43], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(44) = FFN([6], [44], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(45) = FFN([6 7], [45], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(46) = FFN([6], [46], 20, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(47) = FFN([1 2], [47], 3, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(48) = FFN([1 ], [48], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(49) = FFN([6], [49], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1
 pred(50) = FFN([6 7], [50], 5, inputsSet,1);  % good for env4 [s1 m1] -> s1

 
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