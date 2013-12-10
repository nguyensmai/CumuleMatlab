function inputsSet    =  increaseInputsSet(inputsSet, pred, nbFixed, dimM, dimO)

TIME_WINDOW = 1000;
THRES = 0.01;

%intialisation
nPred = numel(pred);
progress  = zeros(1,nPred);
meanError = zeros(1,nPred);
iPred = 1;

while iPred <=numel(pred)
    if (pred(iPred).idFixed == -1) && numel(pred(iPred).sseRec)>4*TIME_WINDOW+1
        meanError(iPred)  = mean(pred(iPred).sseRec(end-TIME_WINDOW:end));
         progress(iPred)   = mean(pred(iPred).sseRec(end-4*TIME_WINDOW:end-3*TIME_WINDOW))...
            - mean(pred(iPred).sseRec(end-TIME_WINDOW:end));
        
        if meanError(iPred) < THRES &&( progress(iPred)>=0)
            pred(iPred).idFixed = nbFixed +1;
            inputsSet = [iputsSet (pred(iPred).idFixed+dimM+dimO)];
        
            if pred(nbFixed+1).idFixed==-1
                predTemp = pred(nbFixed+1);
                pred(nbFixed+1) = pred(iPred);
                pred(iPred)     = predTemp;
            end
            nbFixed = nbFixed +1;
            
        end
        
        
    end
end




end