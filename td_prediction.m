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
tileC = TileCoding(dimTD, [nbTiles nbTiles],[ 0 0], [1 1] ,nbLayers);
tdL = TdLearning(nbTiles^dimTD*nbLayers, 0.1, tileC);
env = Environment(dimO,dimM);
deltaseen = [];
stateseen = [];

%st <- observe state
st   = 2*rand(1,dimO)-1;



while true
    % 1 step
    mt   = zeros(1,2);
    
    % s(t+1) <- observe state
    stp1  = executeAction(env, st, mt);
    % reward <- observe
    rew = stp1(6);
    
    
    % update
    [tdL, delta] = tdUpdate(tdL, st([4 5]), stp1([4 5]), rew);
    stateseen = [stateseen; stp1];
    deltaseen = [deltaseen; delta];
    
    %next time step
    st= stp1;
end



end