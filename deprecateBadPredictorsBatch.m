function [pred nPred mutated, outArchive] = deprecateBadPredictorsBatch(pred, outArchive, inputsSet, dimO, errorL, progressL, time)
%parameters
ARCHIVE_THRES = 0.3;

%intialisation
nPred = numel(pred);

for iDepr = 1:nPred/10
    iPreds = randperm(nPred,2);
    iPred1 = iPreds(1);
    iPred2 = iPreds(2);
    mutated = 0;
    
    %compute fitness of the predictors
    fitness1 = getFitnessBatch( errorL(iPred1), progressL(iPred1) , pred(iPred1).maskOut, outArchive);
    pred(iPred1).quality=fitness1;    
    % archive if good predictors
%     if fitness1>0 && numel(pred(iPred1).sseRec)> 10 && sum(pred(iPred1).sseRec(end-10:end))<ARCHIVE_THRES
%         pred(iPred1).idFixed = time;
%         outArchive = [outArchive; pred(iPred1).maskOut time iPred1]; 
%     end
    
    fitness2 = getFitnessBatch( errorL(iPred2), progressL(iPred2) , pred(iPred2).maskOut, outArchive);
    pred(iPred2).quality=fitness2;
%     if fitness2>0 && numel(pred(iPred2).sseRec)>10 && sum(pred(iPred2).sseRec(end-10:end))<ARCHIVE_THRES
%         pred(iPred2).idFixed = time;
%         outArchive = [outArchive; pred(iPred2).maskOut time iPred2];
%     end
    
    % deprecate based on fitness value
    if (fitness1>fitness2) && (pred(iPred2).idFixed ==-1)
        [pred(iPred2), mutated] = copyAndMutate( pred(iPred1), inputsSet, dimO);
    elseif pred(iPred1).idFixed == -1
        [pred(iPred1), mutated] = copyAndMutate( pred(iPred2), inputsSet, dimO);
    end
    
end

% if numel(pred)>=1
% for iPred = 1:nPred
%     [deprecated pred(iPred)] = deprecateBadPredictor( pred(iPred), memory, time, batch_size);
%     if deprecated ==true
%         if pred(iPred).maskOut==7 || pred(iPred).maskOut==1
% %        if pred(iPred).maskOut==1
%             pred(iPred)
%             disp('what');
%         end
%         maskOut   = pred(iPred).maskOut;
%         probInput = pred(iPred).probInput;
%         probInput(pred(iPred).maskInp) = 0.9*probInput(pred(iPred).maskInp);
%         pred =  pred([1:iPred-1 iPred+1:end]);
%         [pred nPred] = multiplicatePredictors(inputsSet, pred,dimO,dimM, maskOut, probInput);
%     else
%         iPred = iPred +1;
%     end
%     nPred = numel(pred);
% end
% end

end


