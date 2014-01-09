%   ******* CUMULE ALGORITHM ******* %
% Author : Nguyen Sao Mai
% nguyensmai@gmail.com
% nguyensmai.free.fr
%

%% PARAMETERS
nPred = 80;
dimM = 2;
dimO = 8;
MEMORY_SIZE  = 500;
BATCH_SIZE   = 20;

%% 3: INITIALISATION
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
nbFixed = 0;
env = Environment(dimO,dimM);
inputsSet = 1:(dimO+dimM) ;
% 7: initialise short-term memory
sMemory = zeros(MEMORY_SIZE, dimO+dimM+1);
%save test_initialisation_gamma1
time = 1;
outArchive = []; %archive of the outputs of the good predictors
errorLt = [];
progressLt = [];
copyLt=[];
outputsLt ={};
inputsMappingTo = [];
%matlabpool('open',12);


% 4: InitialisenPredpredictors(hand-coded).
pred = initialisePredictors(nPred,inputsSet, env);
% 5: m?randommotorcommand
mt   = env.randomAction;
% 6:	s(t )	?	initial	state.
st   = 2*rand(1,dimO)-1;


%td learning for the errors
% global tdLearner
% dimTD     = 10;
% nbLayers  = 10;
% nbTiles   = 7;
% % -1 <-log10 error<10 (good)
% % 0 (good)<-log10 progress < 6
% tileMin   = [-ones(1,5)   , -ones(1,5)];
% tileMax   = [11*ones(1,5), 7*ones(1,5)];
% tileC     = TileCoding(dimTD, nbTiles*ones(1,dimTD),tileMin,tileMax,nbLayers);
% load tdLearner % to load the fitness function
% % to create a new fitness function
% %tdLearner = TdLearning(nbTiles^dimTD*nbLayers, 0.1, 1, tileC);

%% 9: while true do

while true
    
    %8: LEARNING:
    
    %     10:	for iPred from 1 to nPred do
    %     11:	Compute inDataiPred from sm
    % inDataIpred = computeInput(predIpred, sm);
    
    %         12:	Run the predictor : predDataiPred ? prediPred(inDataiPred)
    % predDataIpred = runPredictor(predIpred,inDataIpred);
    
    %     13:	end for
    
    for t=1:BATCH_SIZE
        %     14:	Execute a motor command m chosen randomly
        mt   = env.randomAction;
        smt = ([st  mt 1]+1)/2;
        sMemory = [sMemory(2:end,:); smt];
        
        %     15:	s(t + 1) ? read from sensorimotor data the new state.
        stp1  = executeAction(env, st, mt);
        
        %     16:	sm(t+1) ? read sensorimotor data
        st  = stp1;
    end
    
    %     17:	(pred, outPred, error, errMap) = TrainPredictors(pred, nPred, predData, sm)
    [pred, outPred, errorL, progressL] = TrainPredictorsBatch(pred, sMemory, BATCH_SIZE, dimO) ;
    errorLt = [errorLt; errorL'];
    progesssLt = [progressLt; progressL'];
    
    
    %% 18:	Neural patterns:
    % 19:	pred = DeprecateBadPredictors(pred, ? error)
    %[outArchive, pred] =  updateArchive(outArchive, pred);
    [pred, nPred, mutated, outArchive] = deprecateBadPredictorsBatch(pred, outArchive, inputsSet, dimO, errorL, progressL,time);

    
    %% post-processing
    if mutated ==1
        copyLt =[copyLt; time];
    end
    
    outputsLt{time} = [];
    for iPred = 1:nPred
        outputsLt{time}{iPred} = pred(iPred).maskOut;
    end
    
    inputsLt{time} = [];
    for iPred = 1:nPred
        inputsLt{time}{iPred} = pred(iPred).maskInp;
    end
    
    for iOut=1:dimO
        for input = 1: numel(inputsSet)
            inputsMappingTo(iOut,input,time) =0;
        end
    end
    
    for iPred=1:nPred
        for iOut=outputsLt{time}{iPred}
            for input = inputsLt{time}{iPred}
                inputsMappingTo(iOut,input,time) =  inputsMappingTo(iOut,input,time)+1;
            end
        end
    end
    
    
    errorPerOutC = cell(nPred);
    errorArchOutC = cell(nPred);
    for iPred=1:nPred
        if ~isempty(pred(iPred).sseRec)
            errorPerOutC{pred(iPred).maskOut} =  [errorPerOutC{pred(iPred).maskOut}; pred(iPred).sseRec(end)];
            if pred(iPred).idFixed>1
                errorArchOutC{pred(iPred).maskOut} =  [errorArchOutC{pred(iPred).maskOut}; pred(iPred).sseRec(end)];
            end
        end
    end
    
    for iDim=1:dimO
            errorPerOut(iDim,time) =  mean(errorPerOutC{iDim});
            nbPerOut(iDim,time) = numel(errorPerOutC{iDim});
            errorArchOut(iDim,time) =  mean(errorArchOutC{iDim});
            nbArchOut(iDim,time) = numel(errorArchOutC{iDim});
   end
    
    
    
    if mod(time,100)==0
        save(['test_learning_',num2str(floor(time/100))])
    end
    time = time + 1;
visualisation_cumuleBatch
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
