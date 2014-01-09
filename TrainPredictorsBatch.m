function [pred, outPred, errorL, progressL] = TrainPredictorsBatch(pred, memory, batch_size, dimO)
% PARAMETERS
NB_EPOCHS = 10;

% INITIALISATION
nPred   = numel(pred);
error   = zeros(nPred,NB_EPOCHS);
outPred = cell(1,nPred);


for iPred = 1:nPred
    %     iPred
    data_in            = memory(end-batch_size-1:end-1, [pred(iPred).maskInp dimO]);
    desired_out        = memory(end-batch_size:end, [pred(iPred).maskOut]);
    desired_out =(desired_out+1)/2;
    
    for iEpoch=1:NB_EPOCHS
        
        [sse pred_out pred(iPred)] = ...
            bkprop(pred(iPred), data_in, desired_out);
        error(iPred, iEpoch)   = error(iPred, iEpoch) + sse;
    end
    error(iPred, iEpoch) = error(iPred, iEpoch)/batch_size;
end


errorL = error(:,end) ; %mean(error,2);
progressL = error(:,1)- error(:,end);


end

