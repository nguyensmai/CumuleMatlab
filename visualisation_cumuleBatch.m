figure(nPred+1)
clf
plot(errorLt);
hold on
semilogy(mean(errorLt,2),'LineWidth',5)
title('error of all predictors and their mean error')
% 
% 
[~, iOut] = max(nbPerOut(:,end));
%for iOut = 1:1 %dimO
    figure(100)
    clf
    b =inputsMappingTo(iOut,:,:);
    b =b(:);
    b = reshape(b,numel(inputsSet),time-1)';
    plot(b)
%     hold on; plot(b(:,1), 'LineWidth',4)
%     hold on; plot(b(:,4), 'LineWidth',4)
%     hold on; plot(b(:,5), 'LineWidth',4)
[~, inp]= sort(b(end,:));
    title(sprintf(['mapping (inputs) to the output',num2str(iOut), '\n <-', num2str(inp(end:-1:end-5))]))
    %end
% 
% 
figure(5)
semilogy(errorPerOut')
title('error per Output')

figure(2)
plot(nbPerOut')
title('nb of predictors per Output')

figure(3)
semilogy(errorArchOut')
title('error of the predictors in the archive')

figure(4)
plot(nbArchOut')
title('nb of the predictors in the archive')



