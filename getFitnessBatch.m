function fitness = getFitnessBatch(pred,iPred, iBest, outArchive)
%
% Author : Nguyen Sao Mai
% nguyensmai@gmail.com
% nguyensmai.free.fr
% 
if isempty(iBest) || numel(pred(iPred).sseRec)>10^3
    fitness = 1;
else
    fitness = pred(iBest).progress/pred(iPred).quality;
end

end