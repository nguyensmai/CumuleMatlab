function [pred nPred mutated, outArchive,globalProbInput] = deprecateBadPredictorsBatch(pred, outArchive, inputsSet, dimO, errorL, progressL, time,globalProbInput)
%parameters
ARCHIVE_THRES = 0.02;
dimInp = numel(inputsSet);

%intialisation
nPred = numel(pred);
already1 = [];
already2 = [];

for iDepr = 1:nPred/10
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
        [ pred, globalProbInput, mutated] = deprecateAlreadyInArchive(pred,iPred2, iPred1, globalProbInput, inputsSet, dimO);
    elseif ~isempty(already1)
        [ pred, globalProbInput, mutated] = deprecateAlreadyInArchive(pred,iPred1, iPred2, globalProbInput, inputsSet, dimO);
    else
        % deprecate based on fitness value
        if (fitness1>fitness2) && (pred(iPred2).idFixed ==-1) && (rand<(fitness1-fitness2)/fitness1)
            [pred, globalProbInput, mutated ] = deprecatedBasedOnFitness(pred, iPred2, iPred1, globalProbInput, inputsSet, dimO );
        elseif pred(iPred1).idFixed == -1 && (rand<(fitness2-fitness1)/fitness2);
            [pred, globalProbInput, mutated ] = deprecatedBasedOnFitness(pred, iPred1, iPred2, globalProbInput, inputsSet, dimO );
        end
    end
end

end

function [ pred, globalProbInput, mutated] = deprecateAlreadyInArchive(pred,iPred2, iPred1, globalProbInput, inputsSet, dimO)
mutated = 0;
if pred(iPred2).idFixed == -1 %iPred2 predicts something already in the archive
    pred(iPred2).maskOut = pred(iPred1).maskOut;
    probInput = mean(globalProbInput([pred(iPred1).maskOut pred(iPred2).maskOut], :));
    [pred(iPred2), mutated] = copyAndMutate( pred(iPred2), inputsSet,dimO,probInput);
else %iPred2 is in the archive
    globalProbInput(pred(iPred2).maskOut, pred(iPred2).maskInp) =  globalProbInput(pred(iPred2).maskOut, pred(iPred2).maskInp)+0.01;
    if rand()< 1-10*pred(iPred1).quality && pred(iPred1).idFixed==-1
        newPred = pred(iPred2);
        newPred.maskOut = pred(iPred1).maskOut;
        probInput = mean(globalProbInput([pred(iPred1).maskOut pred(iPred2).maskOut], :));
        [pred(iPred1), mutated] = copyAndMutate(newPred, inputsSet, dimO,probInput);
    end
end
end

function [pred, globalProbInput, mutated ] = deprecatedBasedOnFitness(pred, iPred1, iPred2, globalProbInput, inputsSet, dimO )
globalProbInput(pred(iPred2).maskOut, pred(iPred2).maskInp) =  globalProbInput(pred(iPred2).maskOut, pred(iPred2).maskInp)+0.01;
globalProbInput(pred(iPred1).maskOut, pred(iPred1).maskInp) =  globalProbInput(pred(iPred1).maskOut, pred(iPred1).maskInp)-0.01;
probInput = mean(globalProbInput([pred(iPred1).maskOut pred(iPred2).maskOut], :));
[pred(iPred1), mutated] = copyAndMutate( pred(iPred2), inputsSet, dimO,probInput);
end
