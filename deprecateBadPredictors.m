function [pred nPred] = deprecateBadPredictors(pred, memory, time, inputsSet,dimO,dimM,timeWindow)
%parameters


%intialisation
nPred = numel(pred);
progress  = zeros(1,nPred);
meanError = zeros(1,nPred);
iPred = 1;

if numel(pred)>=1
    iPred = mod(time,numel(pred))+1;
    %while iPred <=numel(pred)
    [deprecated pred(iPred)] = deprecateBadPredictor( pred(iPred), memory, time, timeWindow);
    if deprecated ==true
        maskOut   = pred(iPred).maskOut;
        probInput = pred(iPred).probInput;
        pred =  pred([1:iPred-1 iPred+1:end]);
        [pred nPred] = multiplicatePredictors(inputsSet, pred,dimO,dimM, maskOut, probInput);
    else
        iPred = iPred +1;
    end
    nPred = numel(pred);
    
end

end


