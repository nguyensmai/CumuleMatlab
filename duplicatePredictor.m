function [pred, inPredi, outPredi] = duplicatePredictor(inputsSet, pred,dimO, dimM, outPredi, probInput)
%parameters
THRES_GOOD_PRED = 0.01;
QUAL_PROG = 0.5;
%initialisation
nPred = numel(pred);
prob  = zeros(1,nPred);
predi = [];
inPredi  = [];
method ='';

if rand()< QUAL_PROG
    for iPred = 1:nPred
        if ~isempty(pred(iPred).meanError) && ... % pred(iPred).meanError<THRES_GOOD_PRED && ...
                ~isempty(pred(iPred).progress)
            prob(iPred) = max(0,pred(iPred).progress);
            %        prob(iPred) = min(1,pred(iPred).quality);
            %prob(iPred) = 1/pred(iPred).meanError;
        else
            prob(iPred) = 0;
        end
    end
    method = ['progress from '];
else
    for iPred = 1:nPred
        if ~isempty(pred(iPred).meanError) && ... % pred(iPred).meanError<THRES_GOOD_PRED && ...
                ~isempty(pred(iPred).quality)
            %        prob(iPred) = pred(iPred).progress;
            prob(iPred)  = 1./max(10^-10, pred(iPred).quality);
            %        prob(iPred) = min(1,pred(iPred).quality);
            %prob(iPred) = 1/pred(iPred).meanError;
        else
            prob(iPred) = 0;
        end
    end
    method = ['quality from '];
end

prob = (prob - min(prob));
prob = prob/sum(prob);

repPred       = find(rand<cumsum(prob),1,'first');

if isempty(repPred)
    repPred = randi(nPred,1);
end
disp('duplicate')
pred(repPred)
inputSize =pred(repPred).sizeInp-1;
inputMask =[];
while (isempty(inputMask) || (size(unique(inputMask),2)~=inputSize) )
    inputMask     = randsample(inputsSet,inputSize, true, probInput);
end
inPredi       = sort(inputMask);

%     outputMask    = randsample([1:dimO],pred(repPred).sizeOut);
%     outPredi      = sort(outputMask);

pred(nPred+1)         = FFN(inPredi, outPredi, pred(repPred).sizeHid1, pred(repPred).sizeHid2, inputsSet);
pred(nPred+1).w1      = pred(repPred).w1;
pred(nPred+1).w2      = pred(repPred).w2;
pred(nPred+1).probInput=probInput;
pred(nPred+1).method  = [method , num2str(pred(repPred).maskInp), ' to ', num2str(pred(repPred).maskOut)];


end