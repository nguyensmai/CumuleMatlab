function fitness = getFitnessBatch(pred,iPred, iBest, outArchive)
%
% Author : Nguyen Sao Mai
% nguyensmai@gmail.com
% nguyensmai.free.fr
% 
if isempty(iBest)
    fitness = 1;
else
    fitness = pred(iBest).quality/pred(iPred).quality;
end

end