function [pred, outPred, errorL] = TrainPredictorsBatch(pred, sMemory, lMemory, batch_size, dimO)
% PARAMETERS
NB_EPOCHS = 10;

% INITIALISATION
nPred   = numel(pred);
error   = zeros(nPred,NB_EPOCHS);
outPred = cell(1,nPred);


for iPred = 1:nPred
    %     iPred
    if batch_size+pred(iPred).delay+1 < size(sMemory,1)
        %desired_out =(desired_out+1)/2;
        for iEpoch=1:NB_EPOCHS
            data_in            = sMemory(end-batch_size-pred(iPred).delay+1:end-pred(iPred).delay, [pred(iPred).maskInp end]);
            desired_out        = sMemory(end-batch_size+1:end, [pred(iPred).maskOut]);
            [sse pred_out pred(iPred)] = ...
                bkprop(pred(iPred), data_in, desired_out);
            %  pred(iPred) = pruning(pred(iPred));
            error(iPred, iEpoch)   =  sse;
        end
        
        inp          = lMemory(end-2*batch_size-pred(iPred).delay+1:end-pred(iPred).delay, [pred(iPred).maskInp end]);
        target       = lMemory(end-2*batch_size+1:end, [pred(iPred).maskOut]);
        output_error = errorInPrediction(pred(iPred),inp, target);
    
        pred(iPred).quality   = output_error + 0.98*pred(iPred).quality;
        pred(iPred).meanError = output_error;

    end
end



errorL = error(:,end) ; %mean(error,2);
%progressL = error(:,1)- error(:,end);


end


function testTrainPredictorsBatch()
% test for 1 predictor
pred         = FFN([2], [1], [5 5], [1], 1);
inputTest = [sort(rand(10,1)) ones(10,1)];
target = cos(3*pi*inputTest(:,1));
test_error = [];
batch_size = 100;
sMemory =[];
y = 0;

while true
    for i=1:batch_size
        mt = rand();
        sMemory = [sMemory; y mt 1];
        y = cos(mt*3*pi);
    end
    
    [pred, outPred, errorL] = TrainPredictorsBatch(pred, sMemory, batch_size);
    test = errorInPrediction(pred,inputTest, target);
    test_error= [test_error; test];
    semilogy(test_error)
    
end

predictedOut =[]
[predictedOut(1), ~]= predict(pred,inputTest(1,:));  %expects 0.5
[predictedOut(2), ~]= predict(pred,inputTest(2,:));  %expects 0.5
[predictedOut(3), ~]= predict(pred,inputTest(3,:));  %expects 1
[predictedOut(4), ~]= predict(pred,inputTest(4,:));  %expects 0
[predictedOut(5), ~]= predict(pred,inputTest(5,:));%expects 0.5
figure; plot([predictedOut;target(1:5)']')



%% small cumule : 1 predictor
nPred = 1;
dimM = 2;
dimO = 1;
MEMORY_SIZE  = 500;
BATCH_SIZE   = 100;
rng('shuffle');
nbFixed = 0;
env = Environment(dimO,dimM);
inputsSet = 1:(dimO+dimM) ;
% 7: initialise short-term memory
sMemory = []; %zeros(MEMORY_SIZE, dimO+dimM+1);
%save test_initialisation_gamma1

% 5: m?randommotorcommand
mt   = env.randomAction;
% 6:	s(t )	?	initial	state.
st   = 2*rand(1,dimO)-1;

pred         = FFN([2], [1], [5 5], [1], 1);

inputTest = [sort(rand(10,1)) ones(10,1)];
target = cos(3*pi*inputTest(:,1));
test_error = [];


while true
    %LEARNING
    for t=1:BATCH_SIZE
        %     14:	Execute a motor command m chosen randomly
        mt   = env.randomAction;
        smt = [st  mt 1];
        sMemory = [sMemory; smt];
       % stp1  = executeAction(env, st, mt);
       stp1 = cos(mt(1)*3*pi);

        st  = stp1;
    end
    [pred, outPred, errorL] = TrainPredictorsBatch(pred, sMemory, BATCH_SIZE, dimO) ;
    test = errorInPrediction(pred,inputTest, target);
    test_error= [test_error; test];
    semilogy(test_error)
end


end
