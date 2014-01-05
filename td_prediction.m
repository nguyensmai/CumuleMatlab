function td_prediction()
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
tdL = TdLearning(nbTiles^dimTD*nbLayers, 0.1,0.9, tileC);
env = Environment(dimO,dimM);
deltaseen = [];
stateseen = [];

%st <- observe state

while true
st   = 2*rand(1,dimO)-1;


for i = 1:1000
%while true
    % 1 step
    mt   = env.randomAction;
    
    % s(t+1) <- observe state
    stp1  = executeAction(env, st, mt);
    % reward <- observe
    rew = (stp1(2)>0);
    
    
    % update
    [tdL, delta] = tdUpdate(tdL, st([1 2]), stp1([1 2]), rew);
    stateseen = [stateseen; stp1];
    deltaseen = [deltaseen; delta];
    
    %next time step
    st= stp1;
end

end

end