%   ******* CUMULE ALGORITHM ******* %
% Author : Nguyen Sao Mai
% nguyensmai@gmail.com
% nguyensmai.free.fr
%

%% PARAMETERS
nPred = 8;
dimM = 2;
dimO = 8;
MEMORY_SIZE  = 500;

%% 3: INITIALISATION
nbFixed = 0;
env = Environment(dimO,dimM);
inputsSet = 1:(dimO+dimM) ;
% 4: InitialisenPredpredictors(hand-coded).
%pred = initialisePredictors(nPred,inputsSet, env);
% 5: m?randommotorcommand
mt   = env.randomAction;
% 6:	s(t )	?	initial	state.
st   = 2*rand(1,dimO)-1;
% 7: initialise short-term memory
sMemory = zeros(MEMORY_SIZE, dimO+dimM+1);
%save test_initialisation_gamma1
time = 1;

%td learning for the errors
global tdLearner
dimTD     = 10;
nbLayers  = 10;
nbTiles   = 7;
% -1 <-log10 error<10 (good)
% 0 (good)<-log10 progress < 6
tileMin   = [-ones(1,5)   , -ones(1,5)];
tileMax   = [11*ones(1,5), 7*ones(1,5)];
tileC     = TileCoding(dimTD, nbTiles*ones(1,dimTD),tileMin,tileMax,nbLayers);
load tdLearner % to load the fitness function 
% to create a new fitness function
%tdLearner = TdLearning(nbTiles^dimTD*nbLayers, 0.1, 1, tileC); 
matlabpool('open',nPred);

pred = initialisePredictors(nPred,inputsSet, env);
%% 9: while true do

while true
    
        %8: LEARNING:
        
        %     10:	for iPred from 1 to nPred do
        %     11:	Compute inDataiPred from sm
        % inDataIpred = computeInput(predIpred, sm);
        
        %         12:	Run the predictor : predDataiPred ? prediPred(inDataiPred)
        % predDataIpred = runPredictor(predIpred,inDataIpred);
        
        %     13:	end for
        
        %     14:	Execute a motor command m chosen randomly
        mt   = env.randomAction;
        smt = ([st  mt 1]+1)/2;
        sMemory = [sMemory(2:end,:); smt];
        
        %     15:	s(t + 1) ? read from sensorimotor data the new state.
        stp1  = executeAction(env, st, mt);
        
        %     16:	sm(t+1) ? read sensorimotor data
        
        
        %     17:	(pred, outPred, error, errMap) = TrainPredictors(pred, nPred, predData, sm)
        [pred, outPred, error] = TrainPredictors(pred, [], smt, stp1 ) ;
        
        
        
        %% 18:	Neural patterns:
        % 19:	pred = DeprecateBadPredictors(pred, ? error)
        [pred nPred] = deprecateBadPredictors(pred, sMemory, time,  inputsSet, dimO, dimM, MEMORY_SIZE);
        %inputsSet    = increaseInputsSet(inputsSet, pred, nbFixed, dimM, dimO);
        
        % 25:	(pred, inPred, outPred ) = multiplicatePredictors(pred, inputsSet)
        %[pred nPred] = multiplicatePredictors(inputsSet, pred,dimO,dimM);
        
        %%
        % 20:	t?t+1
        time = time + 1;
        st  = stp1;
        % 21: endwhile
        %     if mod(time,50000)==0
        %     save(['test_learning_gamma1_',num2str(floor(time/5000))])
        %     end
    if mod(time,200000)==0
    save(['test_learning_',num2str(floor(time/200000))])
    end

end

%% plot results
for iPred = 1:nPred
    iPred
    pred(iPred)
    figure(iPred);
    plot(pred(iPred).sseRec); xlabel('Epochs'); ylabel('Sum squared error (SSE1)'); % The end
end
