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
    if time>timeWindow+1 && numel(pred(iPred).sseRec)>timeWindow+1
        meanError(iPred)  = mean(pred(iPred).sseRec(end-timeWindow:end));
        pred(iPred).meanError = meanError(iPred);
        qualityPredictor2 = qualityError(meanError(iPred)) ;
        
        current_error = zeros(timeWindow-1,1);
        for i=1:timeWindow-1
            data_in          = memory(i,[pred(iPred).maskInp end]);
            desired_out      = memory(i+1,[pred(iPred).maskOut]);
            current_error(i) = errorInPrediction(pred(iPred),data_in, desired_out); 
        end
        progress(iPred)       = meanError(iPred) - mean(current_error);
        pred(iPred).progress  = progress(iPred);
        qualityPredictor1     = qualityProgress(progress(iPred));
        qualityPredictor      = min(1,(qualityPredictor1*qualityPredictor2)^(3./timeWindow));
        pred(iPred).quality   = qualityPredictor;
        r = rand();
        
        if (r>qualityPredictor) && (pred(iPred).idFixed == -1)
            disp(['deprecate predictor ',  num2str(iPred),...
                ' error is ', num2str(meanError(iPred)), ...
                ' progress is ', num2str(progress(iPred)),...
                ' quality is ', num2str(qualityPredictor), ...
                ' at time ', num2str(numel(pred(iPred).sseRec))     ]);
         %   [pred nPred] = multiplicatePredictors(inputsSet, pred,dimO,dimM, pred(iPred).maskOut);
            pred =  pred([1:iPred-1 iPred+1:end]);
        else
%                                     disp(['good predictor ',  num2str(iPred),...
%                                         ' error is ', num2str(meanError(iPred)), ...
%                                         ' progress is ', num2str(progress(iPred)), ...
%                                         ' at quality ', num2str(qualityPredictor)]);
            iPred = iPred +1;
        end
        nPred = numel(pred);
    else
        iPred = iPred +1;
    end
    
end
end


