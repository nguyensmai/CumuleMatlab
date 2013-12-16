classdef TdLearning <handle
   properties
       Vtd %value function
       dimS %dimension of the state space (discretised)
       alpha %learning rate
       gamma % discount rate
       deltaSeen %list of error in prediction
       tileC %tileCoding
   end
   
   methods
       
       function obj = TdLearning(nStates, learn_rate, tileC)
           obj.dimS      = nStates;
           obj.Vtd       = 0.5*ones(1,nStates); 
           obj.alpha     = learn_rate;
           obj.gamma     = 1;
           obj.tileC     = tileC;
           obj.deltaSeen = [];
       end
       
       function val = predict(obj, st)
           val = sum(obj.Vtd(st));
       end
       
       function [obj, delta] = tdUpdate(obj, st, stp1, rew)
           stD     = c2d(obj.tileC, st);
           stp1D   = c2d(obj.tileC, stp1);
           vst     = predict(obj, stD)
           delta   = rew + obj.gamma*predict(obj,stp1D) - vst;
           obj.Vtd(stD) = obj.Vtd(stD) + obj.alpha*( delta );
           obj.deltaSeen = [obj.deltaSeen; delta];
       end
       
   end %end methods
end