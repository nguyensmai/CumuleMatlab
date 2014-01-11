function [pred nPred mutated, outArchive] = deprecateBadPredictorsBatch(pred, outArchive, inputsSet, dimO, errorL, progressL, time)
%parameters
ARCHIVE_THRES = 0.02;
dimInp = numel(inputsSet);

%intialisation
nPred = numel(pred);
already1 = [];
already2 = [];

for iDepr = 1:1 %nPred/10
    iPreds = randperm(nPred,2);
    iPred1 = iPreds(1);
    iPred2 = iPreds(2);
    mutated = 0;
    
    
    % archive if good predictors
    [outArchive, pred,already1] = checkErrorAndAdd(outArchive,pred,iPred1,time);
    [outArchive, pred,already2] = checkErrorAndAdd(outArchive,pred,iPred2,time);
    
    %compute fitness of the predictors
    fitness1 = getFitnessBatch( errorL(iPred1), pred(iPred1).quality, pred(iPred1).maskOut, outArchive);
    fitness2 = getFitnessBatch( errorL(iPred2), pred(iPred2).quality , pred(iPred2).maskOut, outArchive);
    pred(iPred1).quality=fitness1;
    pred(iPred2).quality=fitness2;
    if ~isempty(already2) %out2 is already in the archive
        if pred(iPred2).idFixed == -1 %iPred2 predicts something already in the archive
            pred(iPred2).maskOut = pred(iPred1).maskOut;
            probInput = mean([pred(iPred2).probInput/sum(pred(iPred2).probInput);...
                pred(iPred1).probInput/sum(pred(iPred1).probInput)])+ ...
                0.1*rand(1,dimInp);
            [pred(iPred2), mutated] = copyAndMutate( pred(iPred2), inputsSet,dimO,probInput);
            pred(iPred2).probInput = probInput;
        else %iPred2 is in the archive
            if rand()< 1-10*pred(iPred1).quality && pred(iPred1).idFixed==-1
                newPred = pred(iPred2);
                newPred.maskOut = pred(iPred1).maskOut;
                probInput = mean([pred(iPred2).probInput/sum(pred(iPred2).probInput);...
                    pred(iPred1).probInput/sum(pred(iPred1).probInput)])+ ...
                    0.1*rand(1,dimInp);
                [pred(iPred1), mutated] = copyAndMutate(newPred, inputsSet, dimO,probInput);
                pred(iPred1).probInput = probInput;
            end
        end
    elseif ~isempty(already1)
        if pred(iPred1).idFixed == -1
            pred(iPred1).maskOut = pred(iPred2).maskOut;
            probInput = mean([pred(iPred2).probInput/sum(pred(iPred2).probInput);...
                pred(iPred1).probInput/sum(pred(iPred1).probInput)])+ ...
                0.1*rand(1,dimInp);
            [pred(iPred1), mutated] = copyAndMutate( pred(iPred1), inputsSet, dimO,probInput);
            pred(iPred1).probInput = probInput;
        else
            if rand()< 1-10*pred(iPred2).quality && pred(iPred2).idFixed==-1
                newPred = pred(iPred1);
                newPred.maskOut = pred(iPred2).maskOut;
                probInput = mean([pred(iPred2).probInput/sum(pred(iPred2).probInput);...
                    pred(iPred1).probInput/sum(pred(iPred1).probInput)])+ ...
                    0.1*rand(1,dimInp);
                [pred(iPred2), mutated] = copyAndMutate(newPred, inputsSet, dimO,probInput);
                pred(iPred1).probInput = probInput;
            end
        end
    else
        % deprecate based on fitness value
        if (fitness1>fitness2) && (pred(iPred2).idFixed ==-1)
            probInput = mean([pred(iPred2).probInput/sum(pred(iPred2).probInput);...
                pred(iPred1).probInput/sum(pred(iPred1).probInput)])+ ...
                0.1*rand(1,dimInp);
            probInput(pred(iPred2).maskInp)=probInput(pred(iPred2).maskInp)/2;
            [pred(iPred2), mutated] = copyAndMutate( pred(iPred1), inputsSet, dimO,probInput);
            pred(iPred2).probInput = probInput;
        elseif pred(iPred1).idFixed == -1
            probInput = mean([pred(iPred2).probInput/sum(pred(iPred2).probInput);...
                pred(iPred1).probInput/sum(pred(iPred1).probInput)])+ ...
                0.1*rand(1,dimInp);
            probInput(pred(iPred1).maskInp)=probInput(pred(iPred1).maskInp)/2;
            [pred(iPred1), mutated] = copyAndMutate( pred(iPred2), inputsSet, dimO,probInput);
            pred(iPred1).probInput = probInput;
        end
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


