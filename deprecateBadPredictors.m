function [pred nPred] = deprecateBadPredictors(pred, memory, time, nTime, inputsSet,dimO,dimM,timeWindow)
%parameters


%intialisation
nPred = numel(pred);
progress  = zeros(1,nPred);
meanError = zeros(1,nPred);
iPred = 1;

if size(pred, 2)>=1
  for iTime = 1:nTime
    iPred = mod(time, size(pred, 2))+1;
    %while iPred <=numel(pred)
    [deprecated pred(iTime, iPred)] = deprecateBadPredictor(pred(iTime, iPred), memory, time, iTime, timeWindow);
    if deprecated ==true
        maskOut   = pred(iTime, iPred).maskOut;
        probInput = pred(iTime, iPred).probInput;
        pred =  pred(iTime, [1:iPred-1 iPred+1:end]);
        [pred(iTime) nPred] = multiplicatePredictors(inputsSet, pred(iTime),dimO,dimM, maskOut, probInput);
    else
        iPred = iPred +1;
    end
    nPred = numel(pred);
  end
end
end


