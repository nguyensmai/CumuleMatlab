%   ******* CUMULE ALGORITHM ******* %
% Author : Nguyen Sao Mai
% nguyensmai@gmail.com
% nguyensmai.free.fr
%

%% %%%%%%%%%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%
nPred = 4*50; % 2 layers input mask, 2 layers all 1s , 1 layer input masks, 1 layer all 1s
dimM = 2;
dimO = 50;
MEMORY_SIZE  = 500;
BATCH_SIZE   = 50;

%%  %%%%%%%%%%%%%%% INITIALISATION %%%%%%%%%%%%%%%%%%
rng('shuffle');
nbFixed = 0;
env = Environment(dimO,dimM);
inputsSet = 1:(dimO+dimM) ;
% 7: initialise short-term and long-term memory
sMemory = []; %zeros(MEMORY_SIZE, dimO+dimM+1);
lMemory = [];
%save test_initialisation_gamma1
time = 1;
outArchive = OutputArchive(); %archive of the outputs of the good predictors
errorLt = [];
progressLt = [];
mutateLt=[];
outputsLt ={};
inputsMappingTo = [];
nbArchOut = zeros(dimO,1);
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


%%  %%%%%%%%%%%%%%% RUNNING EVERY TIMESTEP %%%%%%%%%%%%%%%%%%


while true
    
    %% collect data and update the memory
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
    
    if size(lMemory,1)<3*BATCH_SIZE
        lMemory = [sMemory;sMemory;sMemory];
    else
        lMemory=[lMemory(3:end,:); sMemory(randi(BATCH_SIZE,[2 1]),:)];
    end
    
    %% learning
    [pred, outPred, errorL] = TrainPredictorsBatch(pred, sMemory,lMemory, BATCH_SIZE, dimO) ;
    errorLt = [errorLt; errorL'];
    progressL = [];
    
    
    %% Evolution
    mutated = 0;
    [pred, nPred, mutated, outArchive,globalProbInput] = deprecateBadPredictorsBatch(pred, outArchive, inputsSet, dimO, errorL, progressL,time,globalProbInput);
    
   
    %% post-processing
    %{
    nPlot= 10
for i=1:nPlot
subplot(4, nPlot,i)
semilogy(smooth(pred(i).sseRec,10^3))
% ylim([10^-4 2])
end
for i=1:nPlot
subplot(4, nPlot,i+nPlot)
semilogy(smooth(pred(50+i).sseRec,10^3))
% ylim([10^-4 2])
end
for i=1:nPlot
subplot(4, nPlot,i+2*nPlot)
semilogy(smooth(pred(100+i).sseRec,10^3))
% ylim([10^-4 2])
end
    
    for i=1:nPlot
subplot(4, nPlot,i+3*nPlot)
semilogy(smooth(pred(150+i).sseRec,10^3))
% ylim([10^-4 2])
end
    
for i=1:4
subplot(4, nPlot,(i-1)*nPlot+1)
ylim([10^-1 1])
end
for i=1:4
subplot(4, nPlot,(i-1)*nPlot+2)
ylim([10^-6 1])
end
for i=1:4
subplot(4, nPlot,(i-1)*nPlot+3)
ylim([10^-4 1])
end
for i=1:4
subplot(4, nPlot,(i-1)*nPlot+4)
ylim([10^-4 1])
end
for i=1:4
subplot(4, nPlot,(i-1)*nPlot+5)
ylim([10^-4 1])
end
for i=1:4
subplot(4, nPlot,(i-1)*nPlot+6)
ylim([10^-6 1])
end
for i=1:4
subplot(4, nPlot,(i-1)*nPlot+7)
ylim([10^-35 1])
end
for i=1:4
subplot(4, nPlot,(i-1)*nPlot+8)
ylim([10^-5 1])
end
for i=1:4
subplot(4, nPlot,(i-1)*nPlot+9)
ylim([10^-3 1])
end
for i=1:4
subplot(4, nPlot,(i-1)*nPlot+10)
ylim([10^-2 1])
end
    
    
    
    for i=1:5
       subplot(ceil(5/dimO), dimO,i)
       semilogy(smooth(pred(i).sseRec,10^3))
       ylim([10^-6 5])
    end
    %}
    
    
    if mutated ==1
        mutateLt =[mutateLt; time];
    end
    
    outputsLt{time} = [];
    for iPred = 1:nPred
        outputsLt{time}{iPred} = pred(iPred).indOutDelay;
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
            errorPerOutC{pred(iPred).indOutDelay} =  [errorPerOutC{pred(iPred).indOutDelay}; errorL(iPred)];
            if pred(iPred).idFixed>1
                errorArchOutC{pred(iPred).indOutDelay} =  [errorArchOutC{pred(iPred).indOutDelay}; errorL(iPred)];
            end
        end
    end
    
    for iDim=1:dimO
        errorPerOut(iDim,time) =  mean(errorPerOutC{iDim});
        nbPerOut(iDim,time) = numel(errorPerOutC{iDim});
        errorArchOut(iDim,time) =  mean(errorArchOutC{iDim});
        if time>1
            nbArchOut(iDim,time) = nbArchOut(iDim,time-1);
        end
    end
    
    if ~isempty(outArchive.history)
        timeLastArchived = find(outArchive.history(:,4)==time);
        if ~isempty(timeLastArchived)
            lastArchived = outArchive.history(timeLastArchived,:);
            iDim = lastArchived(:,1);
            nbArchOut(iDim,time) = nbArchOut(iDim,time)+1;
        end
    end
    
    time = time + 1;
    if mod(time,100)==0
        save(['environment50bis_',num2str(floor(time/100))])
    end
    visualisation_cumuleBatch
    
    
end



%%  %%%%%%%%%%%%%%% PLOT RESULTS AFTERWARDS %%%%%%%%%%%%%%%%%%


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

% predictions of the archive
figure
archivedL =outArchive.archiveMatrix(:,end);
numPlot = max(20, numel(archivedL));
for i= 1:numPlot
    subplot(4,ceil(numPlot/4),i)
    iPred = archivedL(i);
    inp           = sMemory(end-10-pred(iPred).delay:end-pred(iPred).delay, [pred(iPred).maskInp end]);
    target        = sMemory(end-10:end, [pred(iPred).maskOut]);
    
    output_error = errorInPrediction(pred(iPred),inp, target, 1)
end


figure
for iPred=1:5
    subplot(1,5,iPred)
    inp            = sMemory(end-10-pred(iPred).delay:end-pred(iPred).delay, [pred(iPred).maskInp end]);
    target        = sMemory(end-10:end, [pred(iPred).maskOut]);
    
    output_error = errorInPrediction(pred(iPred),inp, target, 1)
end


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


%time plot of the best predictors
figure
nbArchOutFunc(2,:)=sum(nbArchOut(2:10:end,:),1);
nbArchOutFunc(3,:)=sum(nbArchOut(3:10:end,:),1);
nbArchOutFunc(4,:)=sum(nbArchOut(4:10:end,:),1);
nbArchOutFunc(5,:)=sum(nbArchOut(5:10:end,:),1);
nbArchOutFunc(6,:)=sum(nbArchOut(6:10:end,:),1);
nbArchOutFunc(8,:)=sum(nbArchOut(8:10:end,:),1);
nbArchOutFunc(9,:)=sum(nbArchOut(9:10:end,:),1);
nbArchOutFunc(10,:)=sum(nbArchOut(10:10:end,:),1);
plot(nbArchOutFunc')