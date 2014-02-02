function [pred, nPred, mutated, outArchive,globalProbInput] = addToArchiveBatch(pred, outArchive, inputsSet, dimO, errorL, progressL, time,globalProbInput)
%parameters
ARCHIVE_THRES = 0.001;
dimInp = numel(inputsSet);

%intialisation
nPred = numel(pred);
already1 = [];
already2 = [];
mutated = 0;

for iDepr = 1:1 %nPred/10
    iPreds = randperm(nPred,2);
    iPred1 = iPreds(1);
    iPred2 = iPreds(2);
    mutated1 = 0;
    
    % archive if good predictors
    [outArchive, pred,already1,iPredAlready1] = checkErrorAndAdd(outArchive,pred,iPred1,time);
    [outArchive, pred,already2,iPredAlready2] = checkErrorAndAdd(outArchive,pred,iPred2,time);
    
  
end

end

function globalProbInput = updateGoalProbInput(globalProbInput, pred, iPred)
globalProbInput(pred(iPred).indOutDelay, pred(iPred).maskInp) =  globalProbInput(pred(iPred).indOutDelay, pred(iPred).maskInp)+0.01;
globalProbInput(pred(iPred).indOutDelay, pred(iPred).maskPruned) =  globalProbInput(pred(iPred).indOutDelay, pred(iPred).maskPruned)-0.02;
globalProbInput(pred(iPred).indOutDelay, :) = min(1, max(0.1,globalProbInput(pred(iPred).indOutDelay, :)));
end

function [ pred, globalProbInput, mutated] = deprecateAlreadyInArchive(pred,iArchived, iLooser, globalProbInput, inputsSet, dimO)
mutated = 0;
if pred(iArchived).idFixed == -1 %iArchived predicts something already in the archive
    pred(iArchived).indOutDelay = pred(iLooser).indOutDelay;
    pred(iArchived).maskOut = pred(iLooser).maskOut;
    pred(iArchived).delay = pred(iLooser).delay;
    probInput = mean(globalProbInput([pred(iLooser).indOutDelay pred(iArchived).indOutDelay], :));
    [pred(iArchived), mutated] = copyAndMutate( pred(iArchived), inputsSet,dimO,probInput, 0.1);
    pred(iArchived).method = [pred(iArchived).method, num2str(pred(iLooser).maskInp), ' to ', num2str(pred(iLooser).maskOut)];
else %iArchived is in the archive
    globalProbInput = updateGoalProbInput(globalProbInput, pred, iArchived);
    if rand()< 1-10*pred(iLooser).quality && pred(iLooser).idFixed==-1
        newPred = pred(iArchived);
        newPred.indOutDelay = pred(iLooser).indOutDelay;
        newPred.maskOut = pred(iLooser).maskOut;
        newPred.delay = pred(iLooser).delay;
        probInput = mean(globalProbInput([pred(iLooser).indOutDelay pred(iArchived).indOutDelay], :));
        [pred(iLooser), mutated] = copyAndMutate(newPred, inputsSet, dimO,probInput,0.1);
        pred(iLooser).method = [pred(iLooser).method, num2str(newPred.maskInp), ' to ', num2str(newPred.maskOut)];
    end
end
end

function [pred, globalProbInput, mutated ] = deprecatedBasedOnFitness(pred, iLooser, iWinner, globalProbInput, inputsSet, dimO )
globalProbInput = updateGoalProbInput(globalProbInput, pred, iWinner);
probInput = mean(globalProbInput([pred(iLooser).indOutDelay pred(iWinner).indOutDelay], :));
[pred(iLooser), mutated] = copyAndMutate( pred(iWinner), inputsSet, dimO,probInput,0.5);
pred(iLooser).method = [pred(iLooser).method, num2str(pred(iWinner).maskInp), ' to ', num2str(pred(iWinner).maskOut)];
end


function testUpdateGoalProbInput()
inputsSet = [1:6];
dimO=4;
dimM=2;
globalProbInput=0.4*ones(dimO,dimM+dimO);
disp('reinforce inputmask');
pred(1)=FFN([2 4], [2],5, 5, inputsSet,1);
globalProbInput = updateGoalProbInput(globalProbInput, pred, 1); 
% line 2 : 0.4000    0.4100    0.4000    0.4100    0.4000    0.4000
disp('reinforce inputmask and deprecate maskPruned');
pred(1).maskPruned = [3];
%line 2: 0.4000    0.4200    0.3800    0.4200    0.4000    0.4000
globalProbInput = updateGoalProbInput(globalProbInput, pred, 1)
for i=1:1000
globalProbInput = updateGoalProbInput(globalProbInput, pred, 1);
end
globalProbInput
end

function testDeprecateAlreadyInArchive()
disp(' 2 already predicts something in the archive');
inputsSet = [1:6];
dimO=4;
dimM=2;
globalProbInput=0.4*ones(dimO,dimM+dimO);
pred(1)=FFN([2 4], [2],[5, 5], inputsSet,1);
pred(2)=FFN([1 3], [1],[5, 5], inputsSet,1);
[pred, globalProbInput, mutated] = deprecateAlreadyInArchive(pred,2, 1, globalProbInput, inputsSet, dimO)
globalProbInput % should remain a matrix of 0.4
pred(1) % remains unchanged [2 4] -> 2
pred(2) % pred(2) should have changed to [ 1 3 ] -> 2

disp('2 is already in the archive');
globalProbInput=0.4*ones(dimO,dimM+dimO);
pred(1)=FFN([2 4], [2],5,  inputsSet,1);
pred(2)=FFN([1 3], [1],5,  inputsSet,1);
pred(2).idFixed=1;
[pred, globalProbInput, mutated] = deprecateAlreadyInArchive(pred,2, 1, globalProbInput, inputsSet, dimO)
globalProbInput % line 1 should change :0.4100    0.4000    0.4100    0.4000    0.4000    0.4000
pred(1) %should change to [1 3] ->2
pred(2) %should not change [1 3] ->1

disp('2 is already in the archive and has maskPruned');
globalProbInput=0.4*ones(dimO,dimM+dimO);
pred(1)=FFN([2 4], [2],[5, 5], inputsSet,1);
pred(2)=FFN([1 3], [1],[5, 5], inputsSet,1);
pred(2).idFixed=1;
pred(2).maskPruned = [4];
[pred, globalProbInput, mutated] = deprecateAlreadyInArchive(pred,2, 1, globalProbInput, inputsSet, dimO)
globalProbInput % line 1 should change: 0.4100    0.4000    0.4100    0.3800    0.4000    0.4000 
pred(1) %should change to [1 3] ->2
pred(2) %should not change [1 3] ->1
end

function testDeprecateBasedOnFitness()
inputsSet = [1:6];
dimO=4;
dimM=2;
globalProbInput=0.4*ones(dimO,dimM+dimO);
pred(1)=FFN([2 4], [2],[5, 5], inputsSet,1);
pred(2)=FFN([1 3], [1],[10], inputsSet,1);
pred(2).idFixed=1;
pred(2).maskPruned = [4];
[pred, globalProbInput, mutated ] = deprecatedBasedOnFitness(pred, 1, 2, globalProbInput, inputsSet, dimO )
pred(1) % shoud change [1 3] ->1
pred(2) % should remain the same [1 3] -> 1
globalProbInput %line 1 should change: 0.4100    0.4000    0.4100    0.3800    0.4000    0.4000 
end
