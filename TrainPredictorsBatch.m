function [pred, outPred, errorL, progressL] = TrainPredictorsBatch(pred, memory, batch_size, dimO)
% PARAMETERS
NB_EPOCHS = 10;

% INITIALISATION 
nPred   = numel(pred);
error   = zeros(nPred,NB_EPOCHS);
outPred = cell(1,nPred);


for iPred = 1:nPred
%     iPred 
    for iEpoch=1:NB_EPOCHS
        for t=1:batch_size
            stp1= memory(end-batch_size+t,1:dimO);
            smt = memory(end-batch_size+t-1,:);
            
            data_in{iPred}            = smt([pred(iPred).maskInp end]);
            desired_out{iPred}        = stp1([pred(iPred).maskOut]);
            stp1 = (stp1+1)/2;
            [sse pred_out pred(iPred)] = ...
                bkprop(pred(iPred), data_in{iPred}, desired_out{iPred});
            error(iPred, iEpoch)   = error(iPred, iEpoch) + sse;
        end 
        error(iPred, iEpoch) = error(iPred, iEpoch)/batch_size;
    end   
end

errorL = mean(error,2);
progressL = error(:,1)- error(:,end);


end

