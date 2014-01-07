classdef SARSA
   properties
       Q %value function
       dimS %dimension of the state space (discretised)
       alpha %learning rate
       gamma % discount rate
       deltaSeen %list of error in prediction
       tileS %tileSoding
   end
   
   methods
       
       function obj = TdLearning(nStates, nActions, learn_rate, discount_factor, tileS, tileA)
           obj.dimS      = nStates;
           obj.dimA      = nActions;
           obj.Q         = zeros(nStates, nActions);
           obj.alpha     = learn_rate;
           obj.gamma     = discount_factor;
           obj.tileS     = tileS;
           obj.tileA     = tileA; % 1 dimension tile (discretisation) for a
           obj.deltaSeen = [];
           obj.epsilon   = 0.1;
       end
       
       function val = getQ(obj, st, a)%action:discrete value
           stD     = c2d(obj.tileS, st);
           val = sum(obj.Q(stD, a));
       end
       
       function val = getQD(obj, stD, a)%action:discrete value
           val = sum(obj.Q(stD, a));
       end
       
       function action = chooseAction(obj, stD) %action:discrete value
           if rand()< obj.epsilon
               action = randi(1,obj.dimA); %random action. to change
           else
               val = zeros(1,obj.dimA);
               for iAction =1:obj.dimA
                   val(iAction) = getQD(stD, iAction);
               end
               [~,action] = max(val)
           end
       end
       
       function [vst, obj, delta] = update(obj, st, at, rew, stp1, atp1)
           stD     = c2d(obj.tileS, st);
           stp1D   = c2d(obj.tileS, stp1);
           vst     = getQ(obj, stD, at);
           delta   = rew + obj.gamma*getQ(obj,stp1D, atp1) - vst;
           obj.Q(stD) = obj.Q(stD) + obj.alpha/obj.tileS.nbLayers * delta ;
           obj.deltaSeen = [obj.deltaSeen; delta];
       end
       
   end %end methods
end