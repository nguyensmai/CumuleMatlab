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
    delay = floor((iPred-1)/env.dimO)+1;
    [pred(iPred), inPredi, outPredi] = generatePredictor(inputsSet, pred, env.dimO,out, delay, probInput);
end


end