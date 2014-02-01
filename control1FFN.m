%   ******* FFN ALGORITHM ******* %
% Author : Nguyen Sao Mai
% nguyensmai@gmail.com
% nguyensmai.free.fr
%

% as comparison, a single network as a system

%% PARAMETERS
nPred = 1;
dimM = 2;
dimO = 50;
MEMORY_SIZE  = 500;
BATCH_SIZE   = 100;
NB_EPOCHS = 10;
dtIn  = 0;
dtOut = 0;

%% 3: INITIALISATION

rng('shuffle');
env = Environment(dimO,dimM);
inputsSet = 1:(dimO+dimM) ;
sMemory = []; %zeros(MEMORY_SIZE, dimO+dimM+1);
%save test_initialisation_gamma1
time = 1;
errorLt = [];
data_in2             = [];
data_in              = [];
desired_out2         = [];
desired_out= [];
maskInp = ones(1,(1+dtIn)*numel(inputsSet));
maskOut = ones(1,(1+dtOut)*dimO);

pred         = FFN([1:52], [1:50], [50 50], inputsSet, 1);
% pred.eta =0.001
% pred.alpha = 0.01
% 5: m?randommotorcommand
mt   = env.randomAction;
% 6:	s(t )	?	initial	state.
st   = 2*rand(1,dimO)-1;
iPred =1;

%% 9: while true do
while true
    time =time+1;
    deltas_out = [];
    %LEARNING
    for t=1:BATCH_SIZE
        %     14:	Execute a motor command m chosen randomly
        mt   = env.randomAction;
        smt = [st  mt 1];
        sMemory = [sMemory; smt];
        
        %     15:	s(t + 1) ? read from sensorimotor data the new state.
        stp1  = executeAction(env, st, mt);
        
        %     16:	sm(t+1) ? read sensorimotor data
        st  = stp1;
    end
    
    error   = zeros(nPred,NB_EPOCHS);
    error2  = zeros(nPred,NB_EPOCHS);
    outPred = cell(1,nPred);
    
    if 2*BATCH_SIZE+pred(iPred).delay+1 < size(sMemory,1)
        %desired_out =(desired_out+1)/2;
        for iEpoch=1:NB_EPOCHS
            data_in            = sMemory(end-BATCH_SIZE-pred(iPred).delay:end-pred(iPred).delay, [pred(iPred).maskInp end]);
            desired_out        = sMemory(end-BATCH_SIZE:end, [pred(iPred).maskOut]);
            [sse, pred_out, pred(iPred), deltas_out] = ...
                bkprop(pred(iPred), data_in, desired_out);
            %  pred(iPred) = pruning(pred(iPred));
            error(iPred, iEpoch)   =  sse;
            errorLt = [errorLt;  deltas_out.*deltas_out];
        end
        
        
        %desired_out2        =(desired_out2+1)/2;
        
        for iEpoch=1:NB_EPOCHS
            data_in2            = sMemory(end-2*BATCH_SIZE-pred(iPred).delay+1:end-BATCH_SIZE-pred(iPred).delay, [pred(iPred).maskInp end]);
            desired_out2        = sMemory(end-2*BATCH_SIZE+1:end-BATCH_SIZE, [pred(iPred).maskOut]);
            [sse pred_out pred(iPred)] = ...
                bkprop(pred(iPred), data_in2, desired_out2);
            %pred(iPred) = pruning(pred(iPred));
            error2(iPred, iEpoch)   =  sse;
        end
    end
    
    
    meanError = mean(error,2);
    meanError2 = mean(error2,2);
    pred(iPred).quality   = pred(iPred).meanError-meanError2(iPred) + 0.9*pred(iPred).quality;
    pred(iPred).meanError = meanError(iPred);
    
    errorL = error(:,end) ; %mean(error,2);
     if mod(time,100)==0
    save(['control1FFN5050',num2str(floor(time/100))])
    end    
end

%% plotting
nPlot = ceil(sqrt(dimO));
for i=1:dimO
    subplot(nPlot,nPlot,i)
    semilogy(smooth(abs(errorLt(:,i)),10^3))
    title(['output ',num2str(i)])
    %xlim([0,4*10^4])
end

figure
for t=1:BATCH_SIZE
    %     14:	Execute a motor command m chosen randomly
    mt   = env.randomAction;
    smt = [st  mt 1];
    sMemory = [sMemory; smt];
    
    %     15:	s(t + 1) ? read from sensorimotor data the new state.
    stp1  = executeAction(env, st, mt);
    
    %     16:	sm(t+1) ? read sensorimotor data
    st  = stp1;
end
input            = sMemory(end-20-pred(iPred).delay:end-pred(iPred).delay, [pred(iPred).maskInp end]);
target        = sMemory(end-20:end, [pred(iPred).maskOut]);
    
output_error = errorInPrediction(pred,input, target, [1:50])

window = 100;
meanErrorLt = [];
for i=1:(1+dtOut)*dimO
    meanErrorLt(i,:) = filter(ones(1,window)/window,1, abs(errorLt(:,i)'));
end
plot(meanErrorLt')
