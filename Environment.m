classdef Environment < handle
    
    properties
        hiddenState;
        dimM;
        dimO;
    end
    
    
    
    methods
        
        
        function obj = Environment(dimO,dimM)
            obj.hiddenState = 0;
            obj.dimO = dimO;
            obj.dimM = dimM;
        end
        
        function mt = randomAction(obj)
            mt = 2*rand(1,obj.dimM)-1;
        end

        
        function outputState = executeAction(obj,state, action)
            
            if size(state,2) ~=8
                error(['ENVIRONMENT: error in input state ', num2str(size(state,2)), ' instead of 5']);
            end
            
            if size(action,2) ~=2
                error(['ENVIRONMENT: error in input action', num2str(size(action,2)), ' instead of 2']);
            end
            
            
            outputState(1) = min(1, max(cos(state(1)+action(1)),-1));
            outputState(2) = min(1, max(cos(state(2)+action(2)),-1));
            outputState(3) = min(1, max(2*cos((state(3)^2 + state(4)^2+ state(5)^2)+(action(1)^2 + state(4)^2 +state(5)^2)),-1));
            obj.hiddenState = obj.hiddenState+1;
            outputState(4) =  min(1, max(cos(state(1)+state(2)),-1)); %sin(obj.hiddenState);
            outputState(5) =  min(1, max(cos(action(1)+action(2)),-1)); %sin(obj.hiddenState);
            
            outputState(6) =  min(1, max(state(3)^2 + state(4)^2+ state(5)^2,-1)); 
            outputState(7) =  min(1, max(state(1)^2 + state(2)^2 +state(3)^2,-1)); 
            outputState(8) =  min(1, max(action(1)^2 + state(4)^2 +state(5)^2,-1)); 
            
%             if state(1)+action(1)<0
%                 outputState(5) = 0;
%             elseif state(1)+action(1)<0.8
%                 outputState(5) = state(1)+action(1);
%             else
%                 outputState(5) = state(2)+action(2);
%             end
            
        end %end executeAction
        

        
        
        
    end %end methods
    
    
end