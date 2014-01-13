function [pred, outPred, errorL] = TrainPredictorsBatch(pred, memory, batch_size, dimO)
% PARAMETERS
NB_EPOCHS = 10;
global error

% INITIALISATION
nPred   = numel(pred);
error   = zeros(nPred,NB_EPOCHS);
error2  = zeros(nPred,NB_EPOCHS);
outPred = cell(1,nPred);


for iPred = 1:nPred
    %     iPred
    data_in            = memory(end-batch_size-1:end-1, [pred(iPred).maskInp end]);
    desired_out        = memory(end-batch_size:end, [pred(iPred).maskOut]);
    desired_out =(desired_out+1)/2;
    
    for iEpoch=1:NB_EPOCHS
        
        [sse pred_out pred(iPred)] = ...
            bkprop(pred(iPred), data_in, desired_out);
        error(iPred, iEpoch)   =  sse;
    end
    
    
    data_in2            = memory(end-2*batch_size-1:end-batch_size-1, [pred(iPred).maskInp end]);
    desired_out2        = memory(end-2*batch_size:end-batch_size, [pred(iPred).maskOut]);
    desired_out2        =(desired_out2+1)/2;
    
    for iEpoch=1:NB_EPOCHS
        
        [sse pred_out pred(iPred)] = ...
            bkprop(pred(iPred), data_in2, desired_out2);
        error2(iPred, iEpoch)   =  sse;
    end
end

meanError = mean(error,2);
meanError2 = mean(error2,2);
for iPred=1:nPred
    pred(iPred).quality   = pred(iPred).meanError-meanError2(iPred) ; %+ 0.9*pred(iPred).quality;
    pred(iPred).meanError = meanError(iPred);
end

errorL = error(:,end) ; %mean(error,2);
progressL = error(:,1)- error(:,end);


end

