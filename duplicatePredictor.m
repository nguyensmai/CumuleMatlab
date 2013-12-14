function [pred, inPredi, outPredi] = duplicatePredictor(inputsSet, pred,dimO, dimM, outPredi, probInput)
%parameters
THRES_GOOD_PRED = 0.01;

%initialisation
nPred = numel(pred);
prob  = zeros(1,nPred);
predi = [];
inPredi  = [];

for iPred = 1:nPred
    if ~isempty(pred(iPred).meanError) && ... % pred(iPred).meanError<THRES_GOOD_PRED && ...
            ~isempty(pred(iPred).quality)
        prob(iPred) = min(1,pred(iPred).quality);
        %prob(iPred) = 1/pred(iPred).meanError;
    else
        prob(iPred) = 0;
    end
end
prob = prob/sum(prob);

repPred       = find(rand<cumsum(prob),1,'first');

if isempty(repPred)
  repPred = randi(nPred,1);  
end
    disp('duplicate')
    pred(repPred)
    inputMask     = randsample(inputsSet,pred(repPred).sizeInp-1, true, probInput);
    inPredi       = sort(inputMask);
    
%     outputMask    = randsample([1:dimO],pred(repPred).sizeOut);
%     outPredi      = sort(outputMask);
    
    pred(nPred+1)         = FFN(inPredi, outPredi, pred(repPred).sizeHid1, pred(repPred).sizeHid2, inputsSet);
    pred(nPred+1).w1      = pred(repPred).w1;
    pred(nPred+1).w2      = pred(repPred).w2;
    pred(nPred+1).method  = ['duplicate from ', num2str(pred(repPred).maskInp), ' to ', num2str(pred(repPred).maskOut)];


end