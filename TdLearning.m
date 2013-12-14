classdef TdLearning
   properties
       Vtd %value function
       dimS %dimension of the state space (discretised)
       alpha %learning rate
       gamma % discount rate
   end
   
   methods
       
       function obj = TdLearning(nStates, learn_rate)
           obj.dimS = nStates;
           obj.Vtd = 0.5*ones(1,nStates); 
           obj.alpha = learn_rate;
           obj.gamma = 0.9;
       end
       
       function val = predict(obj, st)
           val = sum(obj.Vtd(st));
       end
       
       function [obj, delta] = tdUpdate(obj, st, stp1, rew)
           vst     = predict(obj, st);
           delta   = rew + obj.gamma*predict(obj,stp1) - vst;
           obj.Vtd(st) = obj.Vtd(st) + obj.alpha*( delta );
       end
       
   end %end methods
end