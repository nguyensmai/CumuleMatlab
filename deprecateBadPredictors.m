function [pred nPred ] = deprecateBadPredictors(pred, memory, time, inputsSet,dimO,dimM,timeWindow)
%parameters


%intialisation
global tdLearner
nPred = numel(pred);
iPred = 1;

if numel(pred)>=1
    iPred = mod(time,numel(pred))+1;
    %while iPred <=numel(pred)
    [deprecated pred(iPred)] = deprecateBadPredictor( pred(iPred), memory, time, timeWindow);
    if deprecated ==true
        if pred(iPred).maskOut==7 || pred(iPred).maskOut==1
%        if pred(iPred).maskOut==1
            pred(iPred)
            disp('what');
        end
        maskOut   = pred(iPred).maskOut;
        probInput = pred(iPred).probInput;
        probInput(pred(iPred).maskInp) = 0.9*probInput(pred(iPred).maskInp);
        pred =  pred([1:iPred-1 iPred+1:end]);
        [pred nPred] = multiplicatePredictors(inputsSet, pred,dimO,dimM, maskOut, probInput);
    else
        iPred = iPred +1;
    end
    nPred = numel(pred);
    
end

end


