function td_prediction()
%parameters
nPred = 8;
dimM = 2;
dimO = 8;
MEMORY_SIZE  = 500;

%initialisation
dimS = 2;
nbLayers = 1;
nbTiles =100;
tileC = TileCoding(dimS, nbTiles,0,1,nbLayers);
tdL = TdLearning(nbTiles^dimS*nbLayers, 0.1);
env = Environment(dimO,dimM);
deltaseen = [];
stateseen = [];

%st <- observe state
st   = 2*rand(1,dimO)-1;

% discretise (tile-coding)
stD = c2d( tileC, st([4 5]) );


while true
    % 1 step
    mt   = zeros(1,2);
    
    % s(t+1) <- observe state
    stp1  = executeAction(env, st, mt);
    % reward <- observe
    rew = stp1(6);
    
    %discretise (tile-coding)
    stp1D = c2d(tileC, stp1([4 5]));
    
    % update
    [tdL, delta] = tdUpdate(tdL, stD, stp1D, rew);
    stateseen = [stateseen; stp1];
    deltaseen = [deltaseen; delta];
    
    %next time step
    st= stp1;
    stD = stp1D;
end



end