function [pred, nPred] = multiplicatePredictors(inputsSet, pred,dimO, dimM, outMask, probInput)
nPred = numel(pred);
p1Thres= 0.3;
p2Thres= 0.7;

%generate random new predictors
% if(numel(pred)<10)
%    p1Thres = p1Thres*100;
%    p2Thres = p2Thres*100;
% end

p1 = rand(1);
if p1<p1Thres
    [pred(nPred+1), inPredi, outPredi] = generatePredictor(inputsSet, pred,dimO, outMask, probInput);
    
    %duplicate existing predictors : to be implemented (TODO)
elseif p1<p2Thres+p1Thres
    [pred, inPredi, outPredi] = duplicatePredictor(inputsSet, pred,dimO,dimM, outMask, probInput);
end



nPred = numel(pred);

end