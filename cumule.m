%   ******* CUMULE ALGORITHM ******* %
% Nguyen Sao Mai
% nguyensmai@gmail.com

%%PARAMETERS
nPred = 8;
nTime = 10;
dimM = 2;
dimO = 8;
MEMORY_SIZE = 500;

%% 3: INITIALISATION
nbFixed = 0;
env = Environment(dimO,dimM);
inputsSet = 1:(dimO+dimM) ;
% 4: InitialisenPredpredictors(hand-coded).
pred = []
for iTime = 1:nTime
  predITime = initialisePredictors(nPred,inputsSet, env, iTime);
  % pred(k) = predictors for t+k
  % pred(k, l) = predictor for state l at time t+k
  pred = [pred predITime]
end
% 5: m?randommotorcommand
mt   = env.randomAction;
% 6:	s(t )	?	initial	state.
st   = 2*rand(1,dimO)-1;
% 7: initialise short-term memory
sMemory = []; %zeros(MEMORY_SIZE, dimO+dimM+1);
save test_initialisation
time = 1;

%% 9: while true do
while true
    %% 8: LEARNING:
    
    %     10:	for iPred from 1 to nPred do
    %     11:	Compute inDataiPred from sm
    % inDataIpred = computeInput(predIpred, sm);
    
    %         12:	Run the predictor : predDataiPred ? prediPred(inDataiPred)
    % predDataIpred = runPredictor(predIpred,inDataIpred);
    
    %     13:	end for
    
    %     14:	Execute a motor command m chosen randomly
    mt   = env.randomAction;
    smt = ([st  mt 1]+1)/2;
    sMemory = [sMemory; smt];

    %     15:	s(t + 1) ? read from sensorimotor data the new state.
    stp1  = executeAction(env, st, mt);
    
    %     16:	sm(t+1) ? read sensorimotor data

	%     17:	(pred, outPred, error, errMap) = TrainPredictors(pred, nPred, predData, sm)
	[pred, outPred, error] = TrainPredictors(pred, [], sMemory, stp1, time);
    
    %% 18:	Neural patterns:
    % 19:	pred = DeprecateBadPredictors(pred, ? error)
    %[pred nPred] = deprecateBadPredictors(pred, sMemory, time, nTime, inputsSet, dimO, dimM, MEMORY_SIZE);
    %inputsSet    = increaseInputsSet(inputsSet, pred, nbFixed, dimM, dimO);
    
    % 25:	(pred, inPred, outPred ) = multiplicatePredictors(pred, inputsSet)
    %[pred nPred] = multiplicatePredictors(inputsSet, pred,dimO,dimM);
    
    %%
    % 20:	t?t+1
    time = time + 1;
    st  = stp1;
    % 21: endwhile
    save(['test_learning',num2str(floor(time/2000))])
end

%% plot results
for iPred = 1:nPred
    iPred
    pred(iPred)
    figure(iPred);
    plot(pred(iPred).sseRec); xlabel('Epochs'); ylabel('Sum squared error (SSE1)'); % The end
end
