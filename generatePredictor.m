function     [predi, inputMask, outputMask] = generatePredictor(inputsSet, pred,dimO, outputMask,delay, probInput)
%parameters
HIDDEN_MAX = floor(numel(inputsSet));

%initialisation
uniqueBool    = 0;
nPred = numel(pred);

while ~uniqueBool
    inputsSetDim  = numel(inputsSet);
    inputSize =-1;
    while inputSize<1 || inputSize>inputsSetDim
        inputSize = round(3*randn(1));
    end
    %inputSize     = randi(inputsSetDim);
    inputMask =[];
    while (isempty(inputMask) || (size(unique(inputMask),2)~=inputSize) ) 
        inputMask     = randsample(inputsSet,inputSize);
    end
    inputMask     = sort(inputMask);
    
    %     outputSize =-1;
    %     while outputSize<1 || outputSize>dimO
    %         outputSize = round(dimO/2*randn(1));
    %     end
    %     %outputSize    = randi(dimO);
    if ~exist('outputMask','var') || isempty(outputMask)        outputSize = 1;
        outputMask    = randsample([1:dimO],outputSize);
        outputMask    = sort(outputMask);
    end
    
    uniqueBool = 1;
    for iPred = 1:nPred
        if isequal([inputMask],[pred(iPred).maskInp])
            if isequal([outputMask],[pred(iPred).maskOut]) && delay ==pred(iPred).delay
                uniqueBool = 0;
                break
            end
        end
    end
    
end

hiddenSize1 =-1;
while hiddenSize1<1 || hiddenSize1>HIDDEN_MAX
    hiddenSize1 = max(2,round(HIDDEN_MAX*randn(1)));
end


predi         = FFN(inputMask, outputMask, hiddenSize1, inputsSet, delay);
predi.probInput = probInput;
predi.method  = ['generated'];
end
