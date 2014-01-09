figure(nPred+1)
clf
plot(errorLt);
hold on
plot(mean(errorLt,2), 'LineWidth',5)
% 
% 
% for iOut = 1:dimO
%     figure(iOut)
%     b =inputsMappingTo(iOut,:,:);
%     b=b(:);
%     b = reshape(b,numel(inputsSet),time)';
%     plot(b)
%     
%     
% end
% 
% 
figure(1)
plot(errorPerOut')

figure(2)
plot(nbPerOut')

% figure(3)
% plot(errorArchOut')
% 
% figure(4)
% plot(nbArchOut')



