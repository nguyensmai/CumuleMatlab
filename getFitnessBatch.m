function fitness = getFitnessBatch(pred,iPred, iBest, outArchive)
%
% Author : Nguyen Sao Mai
% nguyensmai@gmail.com
% nguyensmai.free.fr
% 
if isempty(iBest) || numel(pred(iPred).sseRec)<2000
    fitness = 1;
else
    fitness = max([pred(iPred).progress/pred(iPred).quality  pred(iBest).progress/pred(iPred).quality])^2;
end

end