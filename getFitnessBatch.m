function fitness = getFitnessBatch(error, progress, out, outArchive)

if numel(out)==1 && ~isempty(outArchive) && any(outArchive(:,1)==out)
    fitness = 0;
else
    fitness = 1/error;
end
end