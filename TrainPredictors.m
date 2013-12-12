function [pred, outPred, error] = TrainPredictors(pred, predData, smt, stp1 )
stp1 = (stp1+1)/2;
nPred = numel(pred);
error  = -ones(1,nPred);
outPred = cell(1,nPred);

for iPred = 1:nPred
    data_in   = smt([pred(iPred).maskInp end]);
    desired_out        = stp1([pred(iPred).maskOut]);
    [sse pred_out pred(iPred)] = ...
        bkprop(pred(iPred), data_in, desired_out);
    error(iPred)   = sse;
%     outPred{iPred} =  2*pred_out-1;
%      [sse pred_out pred(iPred)] = ...
%         bkprop(pred(iPred), data_in, desired_out);
%      [sse pred_out pred(iPred)] = ...
%         bkprop(pred(iPred), data_in, desired_out);
%      [sse pred_out pred(iPred)] = ...
%         bkprop(pred(iPred), data_in, desired_out);
end

%     pattern2 = smt(1,[1 3 5]);
%     desired_out2 = desired_out1(:,1)';
%     [sse2 pred_act2 pred(2)] = bkprop(pred(2), pattern2, desired_out2);
%
%     pattern3 = smt(1,[2 4 5]);
%     desired_out3 = desired_out1(:,2)';
%     [sse3 pred_act3 pred(3)] = bkprop(pred(3), pattern3, desired_out3);
%
%     error = [sse sse2 sse3];
%     outPred = [pred_act pred_act2 pred_act3];

end

