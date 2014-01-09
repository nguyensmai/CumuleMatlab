function [outArch  pred]= updateArchive(outArch, pred)
%parameters
ARCHIVE_THRES = 0.3;

i=1;
while i<size(outArch,1)
    iPred1 = outArch(i,3);
    if sum(pred(iPred1).sseRec(end-10:end))>ARCHIVE_THRES
        outArch = outArch([1:i-1 i+1:end],:);
        pred(iPred1).idFixed=-1;
    else
        i=i+1;
    end
    
end