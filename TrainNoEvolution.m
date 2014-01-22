%   ******* CUMULE ALGORITHM ******* %
% Author : Nguyen Sao Mai
% nguyensmai@gmail.com
% nguyensmai.free.fr
%

%% PARAMETERS
nPred = 15;
dimM = 2;
dimO = 5;
MEMORY_SIZE  = 500;
BATCH_SIZE   = 1;

%% 3: INITIALISATION
rng('shuffle');
nbFixed = 0;
env = Environment(dimO,dimM);
inputsSet = 1:(dimO+dimM) ;
% 7: initialise short-term memory
sMemory = []; %zeros(MEMORY_SIZE, dimO+dimM+1);
%save test_initialisation_gamma1
time = 1;
outArchive = OutputArchive(); %archive of the outputs of the good predictors
errorLt = [];
progressLt = [];
mutateLt=[];
outputsLt ={};
inputsMappingTo = [];
nbArchOut = [];
errorArchOut = [];
errorPerOut =  [];
nbPerOut = [];
%matlabpool('open',12);
globalProbInput=0.4*ones(dimO,dimM+dimO);

% 4: InitialisenPredpredictors(hand-coded).
pred = initialisePredictors(nPred,inputsSet, env);
% 5: m?randommotorcommand
mt   = env.randomAction;
% 6:	s(t )	?	initial	state.
st   = 2*rand(1,dimO)-1;


%% 9: while true do

    
while true
    
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
 
    %     17:	(pred, outPred, error, errMap) = TrainPredictors(pred, nPred, predData, sm)
    [pred, outPred, errorL] = TrainPredictorsBatch(pred, sMemory, BATCH_SIZE, dimO) ;
    errorLt = [errorLt; errorL'];
    progressL = [];
    
    
    %% 18:	Neural patterns:
    mutated = 0;
    %[pred, nPred, mutated, outArchive,globalProbInput] = deprecateBadPredictorsBatch(pred, outArchive, inputsSet, dimO, errorL, progressL,time,globalProbInput);
    
    
    %% post-processing
    for i=1:nPred
       subplot(ceil(nPred/dimO), dimO,i)
       semilogy(pred(i).sseRec)
    end
%     if mutated ==1
%         mutateLt =[mutateLt; time];
%     end
%     
%     outputsLt{time} = [];
%     for iPred = 1:nPred
%         outputsLt{time}{iPred} = pred(iPred).indOutDelay;
%     end
%     
%     inputsLt{time} = [];
%     for iPred = 1:nPred
%         inputsLt{time}{iPred} = pred(iPred).maskInp;
%     end
%     
%     for iOut=1:dimO
%         for input = 1: numel(inputsSet)
%             inputsMappingTo(iOut,input,time) =0;
%         end
%     end
%     
%     for iPred=1:nPred
%         for iOut=outputsLt{time}{iPred}
%             for input = inputsLt{time}{iPred}
%                 inputsMappingTo(iOut,input,time) =  inputsMappingTo(iOut,input,time)+1;
%             end
%         end
%     end
%     
%     
%     errorPerOutC = cell(nPred);
%     errorArchOutC = cell(nPred);
%     for iPred=1:nPred
%         if ~isempty(pred(iPred).sseRec)
%             errorPerOutC{pred(iPred).indOutDelay} =  [errorPerOutC{pred(iPred).indOutDelay}; errorL(iPred)];
%             if pred(iPred).idFixed>1
%                 errorArchOutC{pred(iPred).indOutDelay} =  [errorArchOutC{pred(iPred).indOutDelay}; errorL(iPred)];
%             end
%         end
%     end
%     
%     for iDim=1:dimO
%         errorPerOut(iDim,time) =  mean(errorPerOutC{iDim});
%         nbPerOut(iDim,time) = numel(errorPerOutC{iDim});
%         errorArchOut(iDim,time) =  mean(errorArchOutC{iDim});
%         nbArchOut(iDim,time) = numel(errorArchOutC{iDim});
%     end
%     
%     
%     time = time + 1;
%      if mod(time,100)==0
%     save(['environment17',num2str(floor(time/100))])
%     end
%     visualisation_cumuleBatch
end

%% plot results

for iPred=1:nPred
    if pred(iPred).idFixed>1
        iPred
        pred(iPred)
    end
end

for iPred = 1:nPred
    iPred
    pred(iPred)
    figure(iPred);
    plot(pred(iPred).sseRec); xlabel('Epochs'); ylabel('Sum squared error (SSE1)'); % The end
end
