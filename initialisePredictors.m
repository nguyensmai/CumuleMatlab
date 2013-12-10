function pred = initialisePredictors(nPred, inputsSet, env)
% inputsSetDim  = numel(inputsSet);
% inputSize     = randi(inputsSetDim);
% inputMask     = randi(inputsSetDim,1,inputSize);
% inputMask     = sort(unique(inputMask));
% inputSize     = numel(inputMask);
% 
% outputSize    = randi(env.dimO);
% outputMask    = randi(env.dimO,1,outputSize);
% outputMask    = sort(unique(outputMask));
% outputSize    = numel(outputMask);
% 
% pred(1)         = FFN(inputMask, outputMask, 10,10);

% pred(1) = FFN([1 9], [1],5, 5);  % good for env4 [s1 m1] -> s1
% 
% %randomly generated
% for iPred=2:nPred
%     [pred(iPred), inPredi, outPredi] = generatePredictor(inputsSet, pred, dimO, iPred);
% end
% 
% pred(6) = FFN([4 5], [6],5, 5);  % good for env4 [s1 m1] -> s1

% % pre-coded
% iPred = 1;
% for i=1:dimO
%     for j=1:inputsSetDim
%         pred(iPred) = FFN([j], [i], 10,10);
%         iPred = iPred +1;
%     end
% end
%
% for i=1:dimO
%     for j=1:inputsSetDim
%         for k=j:inputsSetDim
%             pred(iPred) = FFN([j k], [i], 10, 10);
%             iPred = iPred +1;
%         end
%     end
% end
%
% for i=1:dimO
%     for j=1:inputsSetDim
%         for k=j:inputsSetDim
%             for l=k:inputsSetDim
%                 pred(iPred) = FFN([j k l], [i], 10, 10);
%                 iPred = iPred +1;
%             end
%         end
%     end
% end
%
% for i=1:dimO
%     for j=1:inputsSetDim
%         for k=j:inputsSetDim
%             for l=k:inputsSetDim
%                 for m=l:inputsSetDim
%                     pred(iPred) = FFN([j k l m], [i], 10, 10);
%                     iPred = iPred +1;
%                 end
%             end
%         end
%     end
% end


% pre-coded
pred(1) = FFN([1 2 6 7], [3],10,10); % good for env4
pred(2) = FFN([1 6], [1],10, 10);  % good for env4 [s1 m1] -> s1
pred(3) = FFN([2 4], [2], 10, 10);
pred(4) = FFN([1 2], [1], 10, 10);
pred(5) = FFN([1 4], [1], 10, 10);
pred(6) = FFN([2 3], [1], 10, 10);
pred(7) = FFN([3 4], [1], 10, 10);
pred(8) = FFN([1 3], [2], 10, 10);
pred(9) = FFN([1 2], [2], 10, 10);
pred(10) = FFN([1 4], [2], 10, 10);
pred(11) = FFN([2 7], [2], 10, 10); %good for env4
pred(12) = FFN([3 4], [2], 10, 10);
pred(13) = FFN([1 2], [4], 10, 10);  %good for env4
pred(14) = FFN([2 4], [4], 10, 10);
pred(15) = FFN([1 2], [5], 10, 10);
pred(16) = FFN([1 4], [5], 10, 10);
pred(17) = FFN([2 3], [5], 10, 10);
pred(18) = FFN([1 2 6 7], [5], 10, 10); %good for env4

end