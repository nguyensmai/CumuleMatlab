figure(nPred+1)
clf
plot(errorLt);
hold on
plot(mean(errorLt,2), 'LineWidth',2)


for iOut = 1:dimO
   figure(iOut+2)
   b =inputsMappingTo(iOut,:,:);
   b=b(:);
   b = reshape(b,numel(inputsSet),time)';
   plot(b)
   
    
end