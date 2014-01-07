% test if the predictors are helpful on a set of goals

%parameters
nbGoals = 1; %number of goals
goalsList = [1]; %goals to define

%parameters
nPred = 8;
dimM = 2;
dimO = 8;
MEMORY_SIZE  = 500;

%initialisation
dimTD = 2;
nbLayers = 1;
nbTiles =10;
tileC = TileCoding(dimTD, [nbTiles nbTiles],[-1 -1], [1 1] ,nbLayers);
env = Environment(dimO,dimM);
deltaseen = zeros(nbGoals,1000);




for iGoal = 1:nbGoals
    sarsa = SARSA(nbTiles^dimTD*nbLayers, 0.1,0.9, tileC);
    goal = goalsList{iGoal};
    stateseen{iGoal} = [];
    st = rand(1,dimO);         %observables
    mt = env.randomAction;     %actions
    xt = zeros(1,dimO+nPred);  %features
    xt(1:dimO) = st;
    
    for t = 1:1000
        %while true
        % 1 step
        
        % s(t+1) <- observe state
        stp1  = executeAction(env, st, mt);
        xtp1(1:dimO) = stp1;

        % reward <- observe
        rew = goal-stp1; %to modify
        
        
        %% use the predictors
        stp1 = (stp1+1)/2;
        for iPred = 1:nPred
            desired_out{iPred}        = stp1([pred(iPred).maskOut]);
            xtp1(dimO+iPred)          = predict(pred(iPred),desired_out{iPred});
        end
        
        action  = chooseAction(sarsa, xtp1);
        mtp1  = 2* action/sarsa.dimA -1;
        
        %% update sarsa
        actiont = floor((mt+1)/sara.dimA);
        actiontp1 = floor((mtp1+1)/sara.dimA);
        [tdL, delta] = update(sarsa, xt, actiont ,rew, xtp1,actiontp1);
        stateseen{iGoal} = [stateseen{iGoal}; stp1];
        deltaseen(iGoal,t) = delta;
        
        %next time step
        st= stp1;
        mt=mtp1;
    end
   
    
    
end


